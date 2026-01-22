/**
 * Loader utility for external HTML partials
 */

/**
 * Load an external HTML partial
 * @param {string} selector - CSS selector for element to replace
 * @param {string} path - Path to the HTML partial file
 * @returns {Promise<void>}
 */
export async function loadPartial(selector, path) {
    try {
        const res = await fetch(path);
        if (!res.ok) {
            throw new Error(`Failed to fetch partial: ${res.status} ${res.statusText}`);
        }
        
        const html = await res.text();
        const range = document.createRange();
        range.selectNode(document.body);
        const fragment = range.createContextualFragment(html);
        
        const element = document.querySelector(selector);
        if (!element) {
            throw new Error(`Element not found: ${selector}`);
        }
        
        element.replaceWith(fragment);
    } catch (error) {
        console.error('Error loading partial:', error);
        throw error;
    }
}

/**
 * Load multiple partials
 * @param {Array} partials - Array of objects with selector and path properties
 * @returns {Promise<void>}
 */
export async function loadPartials(partials) {
    const promises = partials.map(partial => loadPartial(partial.selector, partial.path));
    await Promise.all(promises);
}
