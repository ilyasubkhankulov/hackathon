import fetch from "node-fetch";
import fs from "fs";
import FormData from "form-data";

const uploadAudio = async (audioFilename) => {
  const formData = new FormData();

  formData.append("audio", fs.createReadStream(audioFilename));

  const response = await fetch("https://llava.puzzmatic.com/audio/", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

const uploadImage = async (imageFilename) => {
  const formData = new FormData();

  // formData.append("input_text", "what is going on here?");
  formData.append("image", fs.createReadStream(imageFilename));

  const response = await fetch("https://llava.puzzmatic.com/image/", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

const initiate = async () => {
  const formData = new FormData();

  formData.append("input_text", "I care about initial public offerings");

  const response = await fetch("https://llava.puzzmatic.com/initiate/", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

initiate()
  .then((data) => console.log(data))
  .catch((error) => console.error(error));

// Call the function, replace 'image.png' and 'audio.wav' with your actual file paths
uploadAudio("../files/sample.wav")
  .then((data) => console.log(data))
  .catch((error) => console.error(error));

// Call the function, replace 'image.png' and 'audio.wav' with your actual file paths
uploadImage("../files/sample.png")
  .then((data) => console.log(data))
  .catch((error) => console.error(error));
