import base64
import io
from io import BytesIO
from typing import Annotated

import openai
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
from transformers import TextStreamer

from secret import OPENAI_KEY

openai.api_key = OPENAI_KEY

disable_torch_init()
model_name = "llava-v1.5-13b"
model_path = "liuhaotian/llava-v1.5-7b"
model_base = None
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base, model_name, False, False, device="cuda"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llama_prompt = "Describe what is shown on this video call"

transcript_list = []

image_list = []


def image_to_base64(image: Image):
    output_buffer = BytesIO()
    image.save(output_buffer, format="JPEG")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, image_aspect_ratio):
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
            image = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


@app.get("/hello")
async def hello():
    return "hello"


@app.post("/initiate/")
async def initiate(input_text: Annotated[str, Form()]):
    global importance_prompt
    importance_prompt = input_text
    return {"status": "success"}


@app.post("/audio/")
async def upload_audio(
    audio: UploadFile = File(...),
):
    global transcript_list

    contents = await audio.read()
    audio_file = io.BytesIO(contents)
    audio_file.name = audio.filename
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript.text)
    transcript_list.append(transcript.text)
    print(transcript_list)
    return {"status": "success"}


@app.post("/image/")
async def upload_image(image: UploadFile = File(...)):
    global image_list

    image_contents = await image.read()
    image_file = io.BytesIO(image_contents)
    image = Image.open(image_file).convert("RGB")
    image_list.append(image)

    if len(image_list) >= 3:
        global transcript_list
        transcript_string = ""
        for transcript in transcript_list:
            transcript_string += transcript + " "

        important = determine_importance(transcript_string, importance_prompt)

        image_description = process_image(image, llama_prompt)

        transcript_string = f"Meeting participants are discussing: {transcript_string}. The image from the video call is: {image_description}"

        summary = summarize_transcript(transcript_string)
        transcript_list = []

        image_list = []

        print(
            {
                "status": "display",
                "summary": summary,
                "important": important,
            }
        )
        return {
            "status": "display",
            "summary": summary,
            "important": important,
            "image": image_to_base64(image),
        }

    print(len(image_list))
    return {"status": "success"}


def summarize_transcript(transcript_str):
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {
                "role": "system",
                "content": "You are a diligent and organized professional transcriptionist with an uncanny gift for summarizing complex information. With a background in qualitative research and a knack for synthesizing key points, You utilizes your skills to create pristine summaries from detailed meeting transcripts. You believe in the value of good communication, and your work is aimed at supporting clients in achieving effective and efficient discourse. Your writing style is concise, factual, and clear. You seeks to present the key points of a discussion with precision, ensuring that no essential detail is missed. In your effort to provide a comprehensive and coherent overview, you integrate vital context, highlight decision points, and reflect the tone and dynamics of the discussions. You are quick at identifying recurring themes and standout ideas in a conversation. Given the fast-paced nature of meetings, you service preserves the insights that might otherwise be lost in the moment. As a professional transcriptionist, confidentiality is your paramount principle, ensuring the privacy and trust of all your clients.",
            },
            {"role": "user", "content": transcript_str},
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print(f"Summary: {response.choices[0].message.content}")
    return response.choices[0].message.content


def determine_importance(
    transcript_str,
    importance_prompt,
):
    input = f"Transcript: {transcript_str} \n\n Prompt: {importance_prompt}"
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {
                "role": "system",
                "content": "you will be given an except transcript from a meeting and a user input that represents what the user finds important. You are tasked with determining if the user's defined important subject is  mentioned in the transcript. Your output should be True or False. Nothing else",
            },
            {"role": "user", "content": input},
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print(f"Important: {response.choices[0].message.content}")
    response_bool = bool(response.choices[0].message.content)
    print(f"Important bool: {response_bool}")
    if response_bool:
        return True

    return False


def process_image(image, input_text):
    # image_file: UploadFile, input_text: str
    temperature = 0.2
    max_new_tokens = 512

    # try:
    # Model initialization (move these outside of the endpoint if you want to load the model only once)

    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     "llava-v1.5-13b", None, model_name, False, False, device="cuda"
    # )

    # Your existing code here...

    # Modify the 'image' variable to load the uploaded image
    # image = load_image(image.file)
    # contents = await image.read()

    # image = io.BytesIO(contents)
    # image = Image.open(contents).convert("RGB")
    # image = load_image(image_file)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ("user", "assistant")
    else:
        roles = conv.roles
    # image = load_image(args.image_file)
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, "pad")
    if type(image_tensor) is list:
        image_tensor = [
            image.to(model.device, dtype=torch.float16) for image in image_tensor
        ]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        inp = input_text

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + inp
                )
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        conv.messages[-1][-1] = outputs

        # if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        # return {"response": "Your response goes here"}
        return outputs


@app.post("/upload-everything/")
async def upload_files(
    input_text: Annotated[str, Form()],
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    contents = await audio.read()
    audio_file = io.BytesIO(contents)
    audio_file.name = audio.filename
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript.text)

    # image_file: UploadFile, input_text: str
    temperature = 0.2
    max_new_tokens = 512

    # try:
    # Model initialization (move these outside of the endpoint if you want to load the model only once)

    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     "llava-v1.5-13b", None, model_name, False, False, device="cuda"
    # )

    # Your existing code here...

    # Modify the 'image' variable to load the uploaded image
    # image = load_image(image.file)
    # contents = await image.read()
    image_contents = await image.read()
    image_file = io.BytesIO(image_contents)
    image = Image.open(image_file).convert("RGB")
    # image = io.BytesIO(contents)
    # image = Image.open(contents).convert("RGB")
    # image = load_image(image_file)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ("user", "assistant")
    else:
        roles = conv.roles
    # image = load_image(args.image_file)
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, "pad")
    if type(image_tensor) is list:
        image_tensor = [
            image.to(model.device, dtype=torch.float16) for image in image_tensor
        ]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        inp = input_text

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + inp
                )
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        conv.messages[-1][-1] = outputs

        # if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        # return {"response": "Your response goes here"}
        return {
            "outputs": outputs,
            "transcript": transcript.text,
        }

    # except Exception as e:
    # raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8081, reload=True)
