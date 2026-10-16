const fs = require('fs');
const html = fs.readFileSync('frontend/Chat.html', 'utf8');

// Find all script contents
const regex = /<script\b[^>]*>([\s\S]*?)<\/script>/gi;
let match;
let scriptIndex = 0;
while ((match = regex.exec(html)) !== null) {
  scriptIndex++;
  const code = match[1];
  if (!code.trim()) continue;
  try {
    new Function(code);
    console.log('Script #' + scriptIndex + ' syntax: OK (' + code.length + ' chars)');
  } catch (err) {
    console.error('Script #' + scriptIndex + ' syntax ERROR:', err.message);
  }
}

// Check all document.getElementById calls in the HTML
const idRegex = /document\.getElementById\(['"]([^'"]+)['"]\)/g;
const missingIds = [];
while ((match = idRegex.exec(html)) !== null) {
  const id = match[1];
  if (!html.includes('id="' + id + '"') && !html.includes("id='" + id + "'")) {
    // Some IDs may be created dynamically (e.g. loading-card), check if created dynamically
    if (!html.includes("id: '" + id + "'") && !html.includes('id=\\"' + id + '\\"') && !html.includes('id="' + id + '"')) {
      missingIds.push(id);
    }
  }
}
console.log('Missing element IDs referenced in JS:', missingIds);
