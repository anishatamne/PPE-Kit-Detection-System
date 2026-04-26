import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

export const analyzeImage = async (file) => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("confidence_threshold", 0.5);

    const res = await API.post("/analyze", formData);
    return res.data;
  } catch (err) {
    console.error("API ERROR:", err);
    return null;
  }
};