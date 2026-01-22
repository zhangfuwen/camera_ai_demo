/**
 * Calorie Estimation Controller Component
 * Handles LLM configuration and video analysis
 */

import { createComponent, showElement, hideElement, updateButton, updateStatus, showNotification } from '../lib/componentUtils.js';

export class CalorieEstimationController {
    constructor() {
        this.calorieConfig = {
            llmConfig: {
                baseUrl: '',
                apiKey: '',
                modelName: ''
            }
        };
        this.isConfigOpen = false;
    }

    /**
     * Initialize calorie estimation controller
     */
    initialize() {
        this.loadCalorieConfig();
        this.setupEventListeners();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Configure calories button
        const configBtn = document.getElementById('configure-llm-btn');
        if (configBtn) {
            configBtn.addEventListener('click', () => this.toggleConfigPanel());
        }

        // Save config button
        const saveConfigBtn = document.getElementById('save-llm-config-btn');
        if (saveConfigBtn) {
            console.log("Save config button clicked");
            saveConfigBtn.addEventListener('click', () => this.saveCalorieConfig());
        }


        // Close config panel
        const closeConfigBtn = document.getElementById('close-config-btn');
        if (closeConfigBtn) {
            closeConfigBtn.addEventListener('click', () => this.toggleConfigPanel());
        }
    }

    /**
     * Load calorie configuration from localStorage
     */
    loadCalorieConfig() {
        try {
            const savedConfig = localStorage.getItem('llm-config');
            if (savedConfig) {
                const config = JSON.parse(savedConfig);
                // Only load the LLM config part
                if (config.llmConfig) {
                    this.calorieConfig.llmConfig = config.llmConfig;
                }
            }
        } catch (error) {
            console.error('Error loading LLM config:', error);
        }
    }

    /**
     * Save calorie configuration to localStorage
     */
    saveCalorieConfig() {
        console.log("Save config button clicked");
        try {
            // Collect LLM API configuration
            const baseUrlInput = document.getElementById('llm-base-url');
            const apiKeyInput = document.getElementById('llm-api-key');
            const modelNameInput = document.getElementById('llm-model-name');
            
            if (baseUrlInput && apiKeyInput && modelNameInput) {
                this.calorieConfig.llmConfig.baseUrl = baseUrlInput.value.trim();
                this.calorieConfig.llmConfig.apiKey = apiKeyInput.value.trim();
                this.calorieConfig.llmConfig.modelName = modelNameInput.value.trim();
            }
            
            // Save to localStorage
            localStorage.setItem('llm-config', JSON.stringify(this.calorieConfig));
            
            updateStatus('main-status', 'LLM configuration saved successfully');
            showNotification('LLM configuration saved', 'success');
            
            // Close config panel
            this.toggleConfigPanel();
        } catch (error) {
            console.error('Error saving LLM config:', error);
            updateStatus('main-status', 'Error: Could not save LLM configuration');
            showNotification('Error: Could not save LLM configuration', 'error');
        }
    }

    /**
     * Toggle calorie configuration panel
     */
    toggleConfigPanel() {
        const configPanel = document.getElementById('llm-config-panel');
        if (!configPanel) return;
        
        if (this.isConfigOpen) {
            hideElement(configPanel);
            this.isConfigOpen = false;
        } else {
            this.populateConfigPanel();
            showElement(configPanel);
            this.isConfigOpen = true;
        }
    }

    /**
     * Populate configuration panel with current settings
     */
    populateConfigPanel() {
        const configContent = document.getElementById('llm-config-content');
        if (!configContent) return;
        
        // Clear existing content
        configContent.innerHTML = '';
        
        // Create LLM API Configuration section
        const llmConfigSection = document.createElement('div');
        llmConfigSection.className = 'mb-6 bg-gray-700 p-4 rounded-lg';
        
        // Section header
        const llmHeader = document.createElement('h4');
        llmHeader.className = 'text-lg font-bold mb-3 text-white';
        llmHeader.textContent = 'LLM API Configuration';
        llmConfigSection.appendChild(llmHeader);
        
        // API Base URL
        const baseUrlDiv = document.createElement('div');
        baseUrlDiv.className = 'mb-3';
        
        const baseUrlLabel = document.createElement('label');
        baseUrlLabel.className = 'block text-sm font-medium text-gray-300 mb-1';
        baseUrlLabel.textContent = 'API Base URL';
        baseUrlLabel.setAttribute('for', 'llm-base-url');
        baseUrlDiv.appendChild(baseUrlLabel);
        
        const baseUrlInput = document.createElement('input');
        baseUrlInput.type = 'text';
        baseUrlInput.id = 'llm-base-url';
        baseUrlInput.className = 'w-full bg-gray-800 text-white border border-gray-600 rounded px-3 py-2';
        baseUrlInput.value = this.calorieConfig.llmConfig.baseUrl;
        baseUrlDiv.appendChild(baseUrlInput);
        
        llmConfigSection.appendChild(baseUrlDiv);
        
        // API Key
        const apiKeyDiv = document.createElement('div');
        apiKeyDiv.className = 'mb-3';
        
        const apiKeyLabel = document.createElement('label');
        apiKeyLabel.className = 'block text-sm font-medium text-gray-300 mb-1';
        apiKeyLabel.textContent = 'API Key';
        apiKeyLabel.setAttribute('for', 'llm-api-key');
        apiKeyDiv.appendChild(apiKeyLabel);
        
        const apiKeyInput = document.createElement('input');
        apiKeyInput.type = 'password';
        apiKeyInput.id = 'llm-api-key';
        apiKeyInput.className = 'w-full bg-gray-800 text-white border border-gray-600 rounded px-3 py-2';
        apiKeyInput.value = this.calorieConfig.llmConfig.apiKey;
        apiKeyDiv.appendChild(apiKeyInput);
        
        llmConfigSection.appendChild(apiKeyDiv);
        
        // Model Name
        const modelNameDiv = document.createElement('div');
        modelNameDiv.className = 'mb-3';
        
        const modelNameLabel = document.createElement('label');
        modelNameLabel.className = 'block text-sm font-medium text-gray-300 mb-1';
        modelNameLabel.textContent = 'Model Name';
        modelNameLabel.setAttribute('for', 'llm-model-name');
        modelNameDiv.appendChild(modelNameLabel);
        
        const modelNameInput = document.createElement('input');
        modelNameInput.type = 'text';
        modelNameInput.id = 'llm-model-name';
        modelNameInput.className = 'w-full bg-gray-800 text-white border border-gray-600 rounded px-3 py-2';
        modelNameInput.value = this.calorieConfig.llmConfig.modelName;
        modelNameDiv.appendChild(modelNameInput);
        
        llmConfigSection.appendChild(modelNameDiv);
        
        // Add LLM config section to panel
        configContent.appendChild(llmConfigSection);
    }

    /**
     * Process video with LLM API and display results
     */
    async processVideoWithLLM() {
        // Get the current video file from VideoProcessingController
        const videoFile = this.getCurrentVideoFile();
        if (!videoFile) {
            showNotification('Please select a video file first', 'error');
            return;
        }

        try {
            // Get the LLM client from the global app object
            const llmClient = window.app && window.app.llmClient;
            if (!llmClient) {
                throw new Error('LLM client not available');
            }

            // Test video with LLM client
            const result = await llmClient.testVideoWithLLM(videoFile);

            // Display results using LLM client
            llmClient.displayAnalysisResults(result);

            // Update status
            updateStatus('main-status', 'LLM video analysis completed successfully');
            showNotification('LLM video analysis completed', 'success');
        } catch (error) {
            console.error('Error processing video with LLM:', error);
            updateStatus('main-status', `Error: ${error.message}`);
            showNotification(`Error: ${error.message}`, 'error');
        } finally {
            // Reset button
            if (parseBtn) {
                updateButton(parseBtn, 'LLM Video Test', false);
            }
        }
    }

    /**
     * Get current video file from VideoProcessingController
     * @returns {File|null} - Current video file or null if not found
     */
    getCurrentVideoFile() {
        // Access the video processing controller through the global app object
        if (typeof window.app !== 'undefined' && window.app.videoProcessingController) {
            return window.app.videoProcessingController.currentVideoFile;
        }
        return null;
    }
}
