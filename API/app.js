require('dotenv').config();
const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());

const openaiApiKey = process.env.OPENAI_API_KEY;

app.post('/generate-mnemonic', async (req, res) => {
    const { concept, concept_anchor, components } = req.body;

    try {
        // Generating the analogy using OpenAI's API
        const analogyResponse = await axios.post(
            'https://api.openai.com/v1/completions',
            {
                model: "gpt-3.5-turbo-0125",
                prompt: `Create a detailed analogy to explain the concept '${concept}' using the idea of '${concept_anchor}':`,
                max_tokens: 150
            },
            {
                headers: {
                    'Authorization': `Bearer ${openaiApiKey}`
                }
            }
        );

        // Generating the story for components
        const storyPrompt = components.map(c => `${c.name} (anchor: ${c.anchor}),`).join(' ') + ' using their anchors in a connected narrative.';
        const storyResponse = await axios.post(
            'https://api.openai.com/v1/completions',
            {
                model: "gpt-3.5-turbo-0125",
                prompt: storyPrompt,
                max_tokens: 200
            },
            {
                headers: {
                    'Authorization': `Bearer ${openaiApiKey}`
                }
            }
        );

        res.json({
            analogy: analogyResponse.data.choices[0].text,
            story: storyResponse.data.choices[0].text
        });
    } catch (error) {
        console.error('OpenAI API Error:', error.response ? JSON.stringify(error.response.data) : error.message);
        res.status(500).send('Failed to generate mnemonic');
    }
});

const port = 3000;
app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
