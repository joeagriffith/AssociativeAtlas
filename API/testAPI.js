const axios = require('axios');

const testData = {
  concept: "Newton's Second Law",
  concept_anchor: "tennis ball",
  components: [
    { name: "Force", anchor: "Police" },
    { name: "Mass", anchor: "Your Mum" },
    { name: "Acceleration", anchor: "running" }
  ]
};

axios.post('http://localhost:3000/generate-mnemonic', testData)
  .then(response => {
    console.log('Analogy:', response.data.analogy);
    console.log('Story:', response.data.story);
  })
  .catch(error => {
    console.error('Error making API call:', error.message);
    console.log('Detailed error:', error.response ? error.response.data : 'No additional error data');
  });
