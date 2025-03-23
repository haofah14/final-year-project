import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [prompt, setPrompt] = useState('');
  const [maxLength, setMaxLength] = useState(100);
  const [numSequences, setNumSequences] = useState(1);
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [evaluationScores, setEvaluationScores] = useState([]);

  // Ensure the first letter is capitalized when the component loads or when prompt changes
  useEffect(() => {
    document.title = "Catalyzing Content Generation";
  }, []);

  // Function to capitalize the first letter of a string
  const capitalizeFirstLetter = (string) => {
    if (!string) return string;
    return string.charAt(0).toUpperCase() + string.slice(1);
  };

  // Handler for prompt input changes
  const handlePromptChange = (e) => {
    const inputValue = e.target.value;
    // Automatically capitalize the first letter as user types
    if (inputValue.length > 0) {
      setPrompt(capitalizeFirstLetter(inputValue));
    } else {
      setPrompt(inputValue);
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt.');
      return;
    }

    if (maxLength < 10 || maxLength > 300) {
      setError('Maximum length must be between 10 and 300.');
      return;
    }

    if (numSequences < 1 || numSequences > 5) {
      setError('Number of responses must be between 1 and 5.');
      return;
    }

    // Make sure the prompt starts with a capital letter before submission
    const finalPrompt = capitalizeFirstLetter(prompt.trim());

    setResults([]);
    setEvaluationScores([]);
    setError(null);
    setIsLoading(true);

    try {
      // Generate content from the API
      const response = await fetch('http://localhost:5000/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: finalPrompt,
          max_length: maxLength,
          num_return_sequences: numSequences,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Server responded with status: ${response.status}`);
      }

      const data = await response.json();
      const generatedTexts = data.results;
      setResults(generatedTexts);

      // Now evaluate each generated text
      const evaluations = await Promise.all(
        generatedTexts.map(async (text) => {
          try {
            const evalResponse = await fetch('http://localhost:5000/api/evaluate', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                text: text,
                prompt: finalPrompt,
              }),
            });

            if (!evalResponse.ok) {
              console.error('Evaluation failed for text:', text);
              return null;
            }

            return await evalResponse.json();
          } catch (evalError) {
            console.error('Error evaluating text:', evalError);
            return null;
          }
        })
      );

      // Fixed: Changed parameter name from 'eval' to 'item'
      setEvaluationScores(evaluations.filter(item => item !== null));
    } catch (err) {
      console.error('Error generating content:', err);
      setError(`Failed to generate content: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Helper function to render score with color based on value
  const renderScore = (score) => {
    let color = '#777'; // Default gray
    if (score >= 0.7) color = '#4caf50'; // Green for high scores
    else if (score >= 0.5) color = '#8bc34a'; // Light green
    else if (score >= 0.3) color = '#ffc107'; // Yellow
    else if (score >= 0.1) color = '#ff9800'; // Orange
    else color = '#f44336'; // Red for low scores

    return (
      <span className="score-value" style={{ color }}>
        {(score * 100).toFixed(1)}%
      </span>
    );
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Catalyzing Content Generation</h1>
        <p className="subtitle">enter a prompt, generate creative content, receive a score</p>
      </header>
      
      <main className="app-content">
        <div className="input-section">
          <div className="form-group">
            <label htmlFor="prompt">Enter your prompt:</label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={handlePromptChange}
              placeholder="Type your prompt here..."
              rows="5"
              data-gramm="false"
              data-gramm_editor="false"
              data-enable-grammarly="false"
            />
          </div>

          <div className="controls">
            <div className="form-group">
              <label htmlFor="max-length">Maximum Length:</label>
              <input
                type="number"
                id="max-length"
                value={maxLength}
                onChange={(e) => setMaxLength(parseInt(e.target.value) || 100)}
                min="10"
                max="1000"
              />
              <span className="input-description">maximum length of output to generate</span>
            </div>
            
            <div className="form-group">
              <label htmlFor="num-sequences">Number of Responses:</label>
              <input
                type="number"
                id="num-sequences"
                value={numSequences}
                onChange={(e) => setNumSequences(parseInt(e.target.value) || 1)}
                min="1"
                max="5"
              />
              <span className="input-description">number of different outputs to generate</span>
            </div>
          </div>

          <button 
            className="generate-button" 
            onClick={handleGenerate}
            disabled={isLoading}
          >
            {isLoading ? 'Generating...' : 'Generate Content'}
          </button>
        </div>

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {isLoading && (
          <div className="loading-message">
            <div className="loading-spinner"></div>
            <p>Generating content. This may take a moment...</p>
          </div>
        )}

        {results.length > 0 && (
          <div className="output-section">
            <h2>Generated Content:</h2>
            {results.map((text, index) => (
              <div key={index} className="result-card">
                {results.length > 1 && (
                  <h3>Response {index + 1}</h3>
                )}
                <p>{text}</p>
                
                {/* Creativity Scores Section */}
                {evaluationScores[index] && (
                  <div className="score-section">
                    <h4>Creativity Evaluation</h4>
                    <div className="score-grid">
                      <div className="score-item">
                        <div className="score-label">Fluency:</div>
                        {renderScore(evaluationScores[index].fluency.fluency_score)}
                      </div>
                      <div className="score-item">
                        <div className="score-label">Flexibility:</div>
                        {renderScore(evaluationScores[index].flexibility.flexibility_score)}
                      </div>
                      <div className="score-item">
                        <div className="score-label">Originality:</div>
                        {renderScore(evaluationScores[index].originality.originality_score)}
                      </div>
                      <div className="score-item">
                        <div className="score-label">Elaboration:</div>
                        {renderScore(evaluationScores[index].elaboration.elaboration_score)}
                      </div>
                    </div>
                    <div className="overall-score">
                      <div className="score-label">Overall Creativity:</div>
                      {renderScore(evaluationScores[index].creativity_score)}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;