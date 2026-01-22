/**
 * LLM Client Component
 * Handles client-side LLM functionality for video analysis
 */

import { updateButton, updateStatus, showNotification } from '../lib/componentUtils.js';

export class LLMClient {
    constructor() {
        this.isProcessing = false;
    }

    /**
     * Initialize LLM client
     */
    initialize() {
        // Initialize any client-side LLM resources
        // This could include loading a pre-trained model or setting up API clients
        console.log('LLM Client initialized');
    }

    /**
     * Test video with client-side LLM
     * @param {File} videoFile - Video file to analyze
     * @returns {Promise<Object>} - Analysis results
     */
    async testVideoWithLLM(videoFile) {
        console.log("Testing video with LLM");
        if (!videoFile) {
            throw new Error('No video file provided');
        }

        if (this.isProcessing) {
            throw new Error('LLM processing already in progress');
        }

        this.isProcessing = true;

        try {
            // Get LLM configuration from CalorieEstimationController
            const calorieController = window.app.calorieEstimationController;
            const llmConfig = calorieController.calorieConfig.llmConfig;
            
            // Check if API key is provided
            if (!llmConfig.apiKey) {
                console.log(calorieController.calorieConfig)
                console.log(llmConfig)
                throw new Error('API key is required. Please configure it in the Calorie Estimation settings.');
            }
            
            // Convert video to base64
            const videoBase64 = await this.fileToBase64(videoFile);
            
            // Call OpenAI compatible API
            //return await this.callOpenAIAPI(videoBase64, llmConfig);
            return await this.analyzeVideoBase64(videoBase64, llmConfig);
        } catch (error) {
            console.error('Error analyzing video with LLM:', error);
            throw error;
        } finally {
            this.isProcessing = false;
        }
    }

    /**
     * Convert file to base64
     * @param {File} file - File to convert
     * @returns {Promise<string>} - Base64 representation of the file
     */
    async fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }

    async analyzeVideoBase64(videoBase64, llmConfig) {
        const { baseUrl, apiKey, modelName } = llmConfig;
        const cleanBaseUrl = baseUrl.trim(); // Remove trailing spaces

        const prompt = `Analyze this video and determine if the person is eating. If they are eating, estimate the calories consumed. 
        Please respond with a JSON object in the following format:
        {
            "isEating": true/false,
            "estimatedCalories": number (0 if not eating),
            "additionalInfo": "Detailed description of the video"
        }`;

        // 1. Upload base64 video
        const uploadRes = await fetch(`${cleanBaseUrl}/v1beta/files`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                fileData: {
                    data: videoBase64, // Pure base64 string (no data URI prefix)
                    mimeType: 'video/mp4' // Adjust if different format
                }
            })
        });

        if (!uploadRes.ok) {
            throw new Error(`Upload failed: ${await uploadRes.text()}`);
        }
        const fileData = await uploadRes.json();

        // 2. Analyze with Gemini
        const analyzeRes = await fetch(
            `${cleanBaseUrl}/v1beta/models/${modelName}:generateContent?key=${apiKey}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contents: [{
                        parts: [
                            { text: prompt },
                            {
                                fileData: {
                                    fileUri: fileData.name,
                                    mimeType: fileData.mimeType
                                }
                            }
                        ]
                    }],
                    generationConfig: { maxOutputTokens: 500 }
                })
            }
        );

        if (!analyzeRes.ok) {
            throw new Error(`Analysis failed: ${await analyzeRes.text()}`);
        }

        const result = await analyzeRes.json();
        return result.candidates[0].content.parts[0].text;
    }

    /**
     * Call OpenAI compatible API
     * @param {string} videoBase64 - Base64 encoded video
     * @param {Object} llmConfig - LLM configuration
     * @returns {Promise<Object>} - Analysis results
     */
    async callOpenAIAPI(videoBase64, llmConfig) {
        const prompt = `Analyze this video and determine if the person is eating. If they are eating, estimate the calories consumed. 
        Please respond with a JSON object in the following format:
        {
            "isEating": true/false,
            "estimatedCalories": number (0 if not eating),
            "additionalInfo": "Detailed description of the video"
        }`;

        try {
            console.log("Calling OpenAI API");
            const response = await fetch(`${llmConfig.baseUrl}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${llmConfig.apiKey}`
                },
                body: JSON.stringify({
                    model: llmConfig.modelName,
                    messages: [
                        {
                            role: "user",
                            content: [
                                {
                                    type: "text",
                                    text: prompt
                                },
                                {
                                    type: "image_url",
                                    image_url: {
                                        url: videoBase64
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens: 500
                })
            });
            console.log("Response:", response);

            if (!response.ok) {
                throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            const content = data.choices[0].message.content;

            console.log("Content:", content);
            
            // Try to parse the JSON response
            try {
                return JSON.parse(content);
            } catch (parseError) {
                // If parsing fails, extract information from text
                return this.extractResultsFromText(content);
            }
        } catch (error) {
            console.error('Error calling OpenAI API:', error);
            
            // Fallback to simulation if API call fails
            showNotification('Failed to connect to LLM API. Using simulation instead.', 'warning');
            return await this.simulateLLMAnalysis({ size: 5000000 }); // Simulate with 5MB file size
        }
    }

    /**
     * Extract results from text response
     * @param {string} text - Text response from API
     * @returns {Object} - Parsed results
     */
    extractResultsFromText(text) {
        // Simple extraction logic - in a real implementation, this would be more robust
        const isEating = /eating|food|meal|bite|chew|swallow/i.test(text);
        const calorieMatch = text.match(/(\d+)\s*(?:calories?|kcal)/i);
        const estimatedCalories = calorieMatch ? parseInt(calorieMatch[1]) : (isEating ? 250 : 0);
        
        return {
            isEating,
            estimatedCalories,
            additionalInfo: text
        };
    }

    /**
     * Simulate LLM video analysis (demo purposes)
     * @param {File|Object} videoFile - Video file to analyze or object with size property
     * @returns {Promise<Object>} - Mock analysis results
     */
    async simulateLLMAnalysis(videoFile) {
        // Simulate processing time based on video file size
        const fileSize = videoFile.size || 5000000; // Default to 5MB if size not available
        const processingTime = Math.max(2000, Math.min(10000, fileSize / 1000000 * 2000));
        
        return new Promise((resolve) => {
            setTimeout(() => {
                // Mock results - in a real implementation, these would come from an LLM
                resolve({
                    isEating: Math.random() > 0.3, // 70% chance of detecting eating
                    estimatedCalories: Math.floor(Math.random() * 500) + 100, // 100-600 calories
                    additionalInfo: "This is a mock analysis result. In a real implementation, this would contain detailed analysis from a client-side LLM about the food and eating behavior detected in the video."
                });
            }, processingTime);
        });
    }

    /**
     * Display LLM analysis results
     * @param {Object} result - Analysis results
     */
    displayAnalysisResults(result) {
        // Create results container if it doesn't exist
        let resultsContainer = document.getElementById('llm-test-results');
        if (!resultsContainer) {
            resultsContainer = document.createElement('div');
            resultsContainer.id = 'llm-test-results';
            resultsContainer.className = 'mt-4 bg-gray-800 rounded p-4 max-h-48 overflow-y-auto';
            
            // Find the appropriate place to insert the results
            const videoContainer = document.getElementById('video-container');
            if (videoContainer) {
                videoContainer.insertAdjacentElement('afterend', resultsContainer);
            }
        }

        // Clear existing content
        resultsContainer.innerHTML = '';

        // Create results header
        const header = document.createElement('h3');
        header.className = 'text-lg font-bold mb-3 text-white';
        header.textContent = 'LLM Video Analysis Results';
        resultsContainer.appendChild(header);

        // Display results
        const resultList = document.createElement('ul');
        resultList.className = 'space-y-2';
        
        // Add whether person is eating
        const eatingItem = document.createElement('li');
        eatingItem.className = 'bg-gray-700 p-2 rounded';
        eatingItem.innerHTML = `<strong class="text-green-400">Person Eating:</strong> ${result.isEating ? 'Yes' : 'No'}`;
        resultList.appendChild(eatingItem);
        
        // Add calorie estimate
        const caloriesItem = document.createElement('li');
        caloriesItem.className = 'bg-gray-700 p-2 rounded';
        caloriesItem.innerHTML = `<strong class="text-yellow-400">Estimated Calories:</strong> ${result.estimatedCalories || 0}`;
        resultList.appendChild(caloriesItem);
        
        // Add additional information if available
        if (result.additionalInfo) {
            console.log("additionalInfo:", result.additionalInfo);
            const infoItem = document.createElement('li');
            infoItem.className = 'bg-gray-700 p-2 rounded';
            infoItem.innerHTML = `<strong class="text-blue-400">Additional Info:</strong> ${result.additionalInfo}`;
            resultList.appendChild(infoItem);
        } else {
            console.log("No additional info");
        }
        
        resultsContainer.appendChild(resultList);
        
        // Show the results container
        resultsContainer.style.display = 'block';
    }
}
