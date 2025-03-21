/**
 * Main application entry point for DaVinci AI SEO Tools
 * 
 * This server provides endpoints for various SEO analysis tools
 * managed by specialized agents as described in documentation.
 */

const express = require('express');
const path = require('path');
const logger = require('./utils/logger');
const { validateApiKey } = require('./middleware/auth');
const seoAgents = require('./services/seoAgents');

// Initialize Express application
const app = express();
const PORT = process.env.PORT || 3000;

// Configure middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// Apply API key validation to protected routes
app.use('/api', validateApiKey);

// Home route
app.get('/', (req, res) => {
  res.send(`
    <h1>DaVinci AI SEO Tools API</h1>
    <p>Access our documentation at <a href="/docs">/docs</a> to learn about available endpoints.</p>
  `);
});

// Register SEO analysis endpoints
app.use('/api/on-page', require('./routes/onPageSeo'));
app.use('/api/off-page', require('./routes/offPageSeo'));
app.use('/api/technical', require('./routes/technicalSeo'));
app.use('/api/content', require('./routes/contentSeo'));
app.use('/api/local', require('./routes/localSeo'));
app.use('/api/audit', require('./routes/seoAudit'));

// Documentation endpoint
app.get('/docs', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'docs.html'));
});

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error(`Error: ${err.message}`);
  res.status(err.status || 500).json({
    error: {
      message: err.message || 'Internal Server Error',
      status: err.status || 500
    }
  });
});

// Start the server
app.listen(PORT, () => {
  logger.info(`Server running on port ${PORT}`);
  logger.info(`SEO Agents initialized: ${Object.keys(seoAgents).length} active`);
});

module.exports = app;
