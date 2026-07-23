(function () {
  const config = window.APP_CONFIG;
  const api = config.apiBaseUrl.replace(/\/$/, "");
  const speech = config.speechBaseUrl.replace(/\/$/, "");

  async function request(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    return response;
  }

  window.DeepSeaApi = {
    uploadFrames(formData, sessionId) { return request(`${api}/videoanalyze`, { method: "POST", body: formData, headers: { "X-Session-ID": sessionId } }); },
    askQuestion(formData, sessionId, signal) { return request(`${api}/videoanalyze`, { method: "POST", body: formData, signal, headers: { "X-Session-ID": sessionId } }); },
    getMemos(sessionId) { return request(`${api}/memos`, { headers: { "X-Session-ID": sessionId } }).then(response => response.json()); },
    generateReport(payload) { return request(`${api}/generate_report`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); },
    speechToText(formData) { return request(`${speech}/stt`, { method: "POST", body: formData }).then(response => response.json()); },
    textToSpeech(text) { return request(`${speech}/tts`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) }); },
  };
})();
