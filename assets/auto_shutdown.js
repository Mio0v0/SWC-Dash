// assets/auto_shutdown.js
window.addEventListener("beforeunload", function () {
  try {
    navigator.sendBeacon("/_shutdown", "bye");
  } catch (e) {}
});
