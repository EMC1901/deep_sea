window.DeepSeaDom = {
  setText(element, value) {
    element.textContent = String(value ?? "");
  },
  appendText(parent, className, value) {
    const element = document.createElement("div");
    element.className = className;
    element.textContent = String(value ?? "");
    parent.appendChild(element);
    return element;
  },
  isSafeImageSource(value) {
    return typeof value === "string" && (value.startsWith("data:image/jpeg;base64,") || value.startsWith("https://"));
  },
};
