var canvas = $('#zoom-sdk-video-canvas');

var image = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream");
// here is the most important part because if you dont replace you will get a DOM 18 exception.

window.location.href = image;