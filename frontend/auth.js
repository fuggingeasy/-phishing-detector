const API_BASE_URL = "http://127.0.0.1:8000";
const TOKEN_KEY = "phishing_scanner_token";
const EMAIL_KEY = "phishing_scanner_email";

function saveToken(token, email) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(EMAIL_KEY, email);
}

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function getUserEmail() {
  return localStorage.getItem(EMAIL_KEY);
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(EMAIL_KEY);
}

function logout() {
  clearToken();
  window.location.href = "login.html";
}

/** Redirects to login.html if there is no token. Call at the top of protected pages. */
function requireAuth() {
  if (!getToken()) {
    window.location.href = "login.html";
  }
}

function setBtnLoading(btn, isLoading, label) {
  btn.disabled = isLoading;
  btn.querySelector(".btn__label").textContent = label;
}

/**
 * Generic API helper.
 * @param {string} path - e.g. "/auth/login"
 * @param {string} method - "GET" | "POST"
 * @param {object|null} body - request body, or null
 * @param {boolean} withAuth - attach Authorization header from stored token
 */
async function apiRequest(path, method = "GET", body = null, withAuth = true) {
  const headers = { "Content-Type": "application/json" };
  if (withAuth) {
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (res.status === 401 && withAuth) {
    clearToken();
    window.location.href = "login.html";
    throw new Error("Session expired. Please log in again.");
  }

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    let message = data.detail || `Request failed (${res.status})`;
    if (Array.isArray(data.detail)) {
      // FastAPI validation error format
      message = data.detail.map((d) => d.msg).join(", ");
    }
    throw new Error(message);
  }

  return data;
}
