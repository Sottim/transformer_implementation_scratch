document.getElementById('submitButton').addEventListener('click', async () => {
    const questionInput = document.getElementById('questionInput').value.trim();
    const questionText = document.getElementById('questionText');
    const answerText = document.getElementById('answerText');

    if (!questionInput) {
        alert('Please enter a question!');
        return;
    }

    // Clear previous response
    questionText.textContent = '';
    answerText.textContent = 'Loading...';

    try {
        const response = await fetch('http://localhost:8000/answer/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: questionInput }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        questionText.textContent = `${data.question}`;  // Display the question
        answerText.textContent = `Answer: ${data.answer}`;
    } catch (error) {
        answerText.textContent = `Error: ${error.message}`;
    }
});