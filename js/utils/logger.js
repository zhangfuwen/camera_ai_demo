/**
 * Advanced Logging System for JavaScript
 * Supports multiple log levels with colors and timestamps
 * 
 * Log Levels (same as Python):
 * - VERBOSE (5): Most detailed information
 * - DEBUG (10): Debugging information without lengthy content
 * - INFO (20): Major events and important information
 * - WARNING (30): Warning messages for non-critical issues
 * - ERROR (40): Error messages for critical issues
 */

class Logger {
    constructor(name = 'App', defaultLevel = 'INFO') {
        this.name = name;
        this.defaultLevel = defaultLevel;
        this.levels = {
            VERBOSE: 5,
            DEBUG: 10,
            INFO: 20,
            WARNING: 30,
            ERROR: 40
        };
        
        // Color codes for different log levels
        this.colors = {
            VERBOSE: '#00CED1',      // Dark Turquoise
            DEBUG: '#9370DB',        // Medium Purple
            INFO: '#32CD32',         // Lime Green
            WARNING: '#FFD700',      // Gold
            ERROR: '#FF6347'         // Tomato
        };
        
        // Set default log level from environment or parameter
        this.currentLevel = this.levels[defaultLevel] || this.levels.INFO;
        
        // Check for environment variable (like in Python)
        const envLevel = this.getLogLevelFromEnv();
        if (envLevel) {
            this.currentLevel = envLevel;
        }
    }
    
    getLogLevelFromEnv() {
        // Try to get log level from various sources
        if (typeof window !== 'undefined') {
            const level = window.localStorage.getItem('LOG_LEVEL') || 
                         window.sessionStorage.getItem('LOG_LEVEL');
            if (level && this.levels[level.toUpperCase()]) {
                return this.levels[level.toUpperCase()];
            }
        }
        return null;
    }
    
    shouldLog(level) {
        return level >= this.currentLevel;
    }
    
    formatMessage(level, levelName, message, callerInfo) {
        const timestamp = new Date().toISOString();
        const color = this.colors[levelName];
        
        // Get caller information if available
        let location = '';
        if (callerInfo) {
            location = `[${callerInfo.functionName}:${callerInfo.lineNumber}]`;
        } else {
            // Fallback to simple line number detection
            const stack = new Error().stack;
            if (stack) {
                const lines = stack.split('\n');
                // Skip the current function and find the actual caller
                for (let i = 3; i < lines.length; i++) {
                    const line = lines[i];
                    if (line && !line.includes('logger.js') && !line.includes('Logger.')) {
                        const match = line.match(/at\s+(.+?)\s+\((.+?):(\d+):\d+\)/);
                        if (match) {
                            location = `[${match[1]}:${match[3]}]`;
                            break;
                        }
                        // Fallback for different browser formats
                        const fallbackMatch = line.match(/(.+?)@(.+?):(\d+)/);
                        if (fallbackMatch) {
                            location = `[${fallbackMatch[1]}:${fallbackMatch[3]}]`;
                            break;
                        }
                    }
                }
            }
        }
        
        return `%c[${levelName.padEnd(8)}][${timestamp}]${location} ${message}`;
    }
    
    log(level, levelName, message, ...args) {
        if (!this.shouldLog(level)) {
            return;
        }
        
        const color = this.colors[levelName];
        
        // Capture caller information
        const stack = new Error().stack;
        let callerInfo = null;
        
        if (stack) {
            const lines = stack.split('\n');
            // Skip the current function and find the actual caller
            for (let i = 3; i < lines.length; i++) {
                const line = lines[i];
                if (line && !line.includes('logger.js') && !line.includes('Logger.')) {
                    const match = line.match(/at\s+(.+?)\s+\((.+?):(\d+):\d+\)/);
                    if (match) {
                        callerInfo = {
                            functionName: match[1],
                            filename: match[2].split('/').pop(),
                            lineNumber: match[3]
                        };
                        break;
                    }
                    // Fallback for different browser formats
                    const fallbackMatch = line.match(/(.+?)@(.+?):(\d+)/);
                    if (fallbackMatch) {
                        callerInfo = {
                            functionName: fallbackMatch[1],
                            filename: fallbackMatch[2].split('/').pop(),
                            lineNumber: fallbackMatch[3]
                        };
                        break;
                    }
                }
            }
        }
        
        const formattedMessage = this.formatMessage(level, levelName, message, callerInfo);
        
        // Use appropriate console method based on level
        switch (levelName) {
            case 'ERROR':
                console.error(formattedMessage, `color: ${color}`, ...args);
                break;
            case 'WARNING':
                console.warn(formattedMessage, `color: ${color}`, ...args);
                break;
            default:
                console.log(formattedMessage, `color: ${color}`, ...args);
        }
    }
    
    // Convenience methods
    verbose(message, ...args) {
        this.log(this.levels.VERBOSE, 'VERBOSE', message, ...args);
    }
    
    debug(message, ...args) {
        this.log(this.levels.DEBUG, 'DEBUG', message, ...args);
    }
    
    info(message, ...args) {
        this.log(this.levels.INFO, 'INFO', message, ...args);
    }
    
    warning(message, ...args) {
        this.log(this.levels.WARNING, 'WARNING', message, ...args);
    }
    
    error(message, ...args) {
        this.log(this.levels.ERROR, 'ERROR', message, ...args);
    }
    
    // Method to set log level dynamically
    setLevel(level) {
        if (typeof level === 'string') {
            this.currentLevel = this.levels[level.toUpperCase()] || this.levels.INFO;
        } else {
            this.currentLevel = level;
        }
    }
    
    // Method to get current level
    getLevel() {
        for (const [name, value] of Object.entries(this.levels)) {
            if (value === this.currentLevel) {
                return name;
            }
        }
        return 'UNKNOWN';
    }
}

// Create a global logger instance
const globalLogger = new Logger('Global', 'INFO');

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Logger, globalLogger };
} else if (typeof window !== 'undefined') {
    window.Logger = Logger;
    window.logger = globalLogger;
}

// Also provide a simple export for ES6 modules
export { Logger, globalLogger };
