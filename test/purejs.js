const uploadFiles = async (imageFile, audioFile) => {
  const formData = new FormData();

  formData.append("image", imageFile);
  formData.append("audio", audioFile);

  const response = await fetch("https://llava.puzzmatic.com/upload/", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
};

// Call the function, replace 'image.png' and 'audio.wav' with your actual file objects
uploadFiles("../files/sample.png", "../files/sample.wav")
  .then((data) => {
    console.log(data); // JSON from `response.json()` call
  })
  .catch((error) => console.error(error));
