const client = ZoomMtgEmbedded.createClient();

let meetingSDKElement = document.getElementById("meetingSDKElement");

function generateSignature(key, secret, meetingNumber, role) {
  var iat = Math.round(new Date().getTime() / 1000) - 30;
  var exp = iat + 60 * 60 * 2;
  var oHeader = { alg: "HS256", typ: "JWT" };

  var oPayload = {
    sdkKey: key,
    appKey: key,
    mn: meetingNumber,
    role: role,
    iat: iat,
    exp: exp,
    tokenExp: exp,
  };

  var sHeader = JSON.stringify(oHeader);
  var sPayload = JSON.stringify(oPayload);
  var sdkJWT = KJUR.jws.JWS.sign("HS256", sHeader, sPayload, secret);
  return sdkJWT;
}

var sig = generateSignature(
  "koXQzo91RFWgqls3gvfVpA",
  "4hTae08ojOTULZYHIQnuWRJe7fIaBbFm",
  6765551127,
  0
);
console.log(sig);

client.init({ zoomAppRoot: meetingSDKElement, language: "en-US" });

client.join({
  sdkKey: "koXQzo91RFWgqls3gvfVpA",
  signature: sig,
  meetingNumber: "6765551127",
  password: "eWllTkN1NWxyUnE3Z1NoeGFnL005dz09",
  userName: "ateam_bot",
});
