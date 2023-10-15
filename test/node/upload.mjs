import fetch from "node-fetch";
import fs from "fs";
import FormData from "form-data";

const uploadFiles = async (imageFilename, audioFilename) => {
  const formData = new FormData();

  formData.append("input_text", "what is going on here?");
  formData.append("image", fs.createReadStream(imageFilename));
  formData.append("audio", fs.createReadStream(audioFilename));

  const response = await fetch("https://llava.puzzmatic.com/upload/", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

// Call the function, replace 'image.png' and 'audio.wav' with your actual file paths
uploadFiles("../files/sample.png", "../files/sample.wav")
  .then((data) => console.log(data))
  .catch((error) => console.error(error));
