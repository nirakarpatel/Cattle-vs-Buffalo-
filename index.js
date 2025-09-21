// index.js

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

// --- Configuration ---
const PORT = process.env.PORT || 3001;
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://127.0.0.1:5000/predict';

// --- App Initialization ---
const app = express();
app.use(cors());

// --- Middleware ---
const upload = multer({ storage: multer.memoryStorage() });

// --- Route Handler ---
const handleClassification = async (req, res) => {
    const startTime = Date.now();

    if (!req.file) {
        return res.status(400).json({ error: "No image file provided." });
    }

    try {
        const form = new FormData();
        form.append('file', req.file.buffer, req.file.originalname);

        console.log(`Forwarding image to Python API at ${PYTHON_API_URL}...`);

        const response = await axios.post(PYTHON_API_URL, form, {
            headers: { ...form.getHeaders() },
        });
        
        const duration = Date.now() - startTime;
        console.log('✅ Response from Python API:', response.data);
        console.log(`⏱️  Request processed in ${duration} ms`);

        res.json(response.data);

    } catch (error) {
        const duration = Date.now() - startTime;
        logApiError(error, duration);
        
        if (error.response) {
            res.status(error.response.status).json({
                error: "Prediction service returned an error.",
                details: error.response.data,
            });
        } else if (error.request) {
            res.status(503).json({ error: "Prediction service is unavailable." });
        } else {
            res.status(500).json({ error: 'An internal server error occurred.' });
        }
    }
};

// --- Error Logging Helper ---
const logApiError = (error, duration) => {
    if (error.response) {
        console.error(`❌ Error from Python API (Status: ${error.response.status}):`, error.response.data);
    } else if (error.request) {
        console.error('❌ No response from Python API. Is the server running?');
    } else {
        console.error('❌ Internal Server Error:', error.message);
    }
    console.error(`⏱️  Request failed after ${duration} ms`);
};

// --- Routes ---
app.post('/api/classify', upload.single('image'), handleClassification);

// --- Server Start ---
app.listen(PORT, () => {
    console.log(`✅ Server running at http://localhost:${PORT}`);
    console.log(`--> Forwarding requests to: ${PYTHON_API_URL}`);
});