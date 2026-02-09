import axios from "axios";

// 🔥 Use direct backend URL (Flask)
const BASE_URL = "http://127.0.0.1:5000";

export const getCities = async () => {
  return axios.get(`${BASE_URL}/api/cities`);
};

export const getForecast = async (city) => {
  return axios.post(`${BASE_URL}/api/forecast`, { city });
};
