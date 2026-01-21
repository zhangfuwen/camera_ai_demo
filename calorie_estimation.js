class CalorieEstimation {
  constructor(options = {}) {
    this.apiKey = options.apiKey || '';
    this.baseUrl = options.baseUrl || '';
    this.frameRate = options.frameRate || 2000; // Default to 2 seconds
    this.onCalorieUpdate = options.onCalorieUpdate || (() => {});
    this.onError = options.onError || (() => {});
    this.isEstimating = false;
    this.estimationInterval = null;
    this.totalCalories = 0;
    this.eatenFoods = []; // Track foods that have been eaten
  }

  updateConfig(config) {
    if (config.apiKey !== undefined) this.apiKey = config.apiKey;
    if (config.baseUrl !== undefined) this.baseUrl = config.baseUrl;
    if (config.frameRate !== undefined) this.frameRate = config.frameRate;
    
    // If currently running, restart with new config
    if (this.isEstimating) {
      this.stopEstimation();
      this.startEstimation(this.currentVideoElement);
    }
  }

  async estimateCaloriesFromFrame(videoElement) {
    // Create a temporary canvas to capture the frame
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    // Convert to data URL
    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);

    // Extract base64 image data
    const base64Image = imageDataUrl.split(',')[1];

    try {
      // Prepare the request for OpenAI-compatible API
      const requestBody = {
        model: "gemini-2.5-flash-preview-09-2025", // Adjust model name as needed for your API
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Analyze this image and determine if a person is eating food. If they are, identify the food being consumed and estimate the calories per mouthful. Respond in JSON format with the following structure: {isEating: boolean, caloriesPerMouthful: number, foodType: string, confidence: number}. If no eating is detected, set caloriesPerMouthful to 0."
              },
              {
                type: "image_url",
                image_url: {
                  url: `data:image/jpeg;base64,${base64Image}`
                }
              }
            ]
          }
        ],
        temperature: 0.2,
        max_tokens: 500
      };

      const response = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();
      
      // Parse the response - different OpenAI-compatible APIs might have slightly different structures
      let content = '';
      if (data.choices && data.choices[0]) {
        if (data.choices[0].message) {
          content = data.choices[0].message.content;
        } else if (data.choices[0].delta) {
          content = data.choices[0].delta.content;
        }
      }
      
      // Try to parse the JSON response from the model
      let parsedResult;
      try {
        // Look for JSON within the response content
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          parsedResult = JSON.parse(jsonMatch[0]);
        } else {
          // If no explicit JSON found, try to parse the whole content
          parsedResult = JSON.parse(content);
        }
      } catch (parseError) {
        console.warn('Could not parse JSON from response:', content);
        // Provide default values if JSON parsing fails
        parsedResult = {
          isEating: false,
          caloriesPerMouthful: 0,
          foodType: 'unknown',
          confidence: 0
        };
      }

      return parsedResult;
    } catch (error) {
      console.error('Error during calorie estimation:', error);
      this.onError(error);
      return {
        isEating: false,
        caloriesPerMouthful: 0,
        foodType: 'unknown',
        confidence: 0
      };
    }
  }

  async handleEstimationCycle(videoElement) {
    try {
      const result = await this.estimateCaloriesFromFrame(videoElement);
      
      if (result.isEating && result.caloriesPerMouthful > 0) {
        // Add to total calories
        this.totalCalories += result.caloriesPerMouthful;
        
        // Store the eaten food for tracking
        this.eatenFoods.push({
          calories: result.caloriesPerMouthful,
          foodType: result.foodType,
          timestamp: Date.now()
        });
      }
      
      // Call the update callback with current data
      this.onCalorieUpdate({
        totalEstimatedCalories: this.totalCalories,
        lastDetection: result,
        eatenFoodsCount: this.eatenFoods.length
      });
    } catch (error) {
      console.error('Error in estimation cycle:', error);
      this.onError(error);
    }
  }

  startEstimation(videoElement) {
    if (this.isEstimating) {
      console.warn('Calorie estimation already running');
      return;
    }
    
    if (!this.apiKey || !this.baseUrl) {
      throw new Error('API key and base URL must be set before starting estimation');
    }
    
    this.isEstimating = true;
    this.currentVideoElement = videoElement;
    
    // Start the periodic estimation
    this.estimationInterval = setInterval(() => {
      this.handleEstimationCycle(videoElement);
    }, this.frameRate);
    
    console.log(`Started calorie estimation with frame rate: ${this.frameRate}ms`);
  }

  stopEstimation() {
    if (!this.isEstimating) {
      console.warn('Calorie estimation is not running');
      return;
    }
    
    clearInterval(this.estimationInterval);
    this.estimationInterval = null;
    this.isEstimating = false;
    this.currentVideoElement = null;
    
    console.log('Stopped calorie estimation');
  }

  isRunning() {
    return this.isEstimating;
  }

  resetTracking() {
    this.totalCalories = 0;
    this.eatenFoods = [];
    this.onCalorieUpdate({
      totalEstimatedCalories: this.totalCalories,
      eatenFoodsCount: this.eatenFoods.length
    });
  }

  getTotalCalories() {
    return this.totalCalories;
  }
}
