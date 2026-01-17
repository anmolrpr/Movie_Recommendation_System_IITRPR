const API_URL = "";

// State
let token = localStorage.getItem("access_token");
let currentMovieId = null;

// Init
if (token) {
    showDashboard();
    fetchUser();
} else {
    showAuthForm('login');
}

// Auth Functions
function showAuthForm(type) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('login-form').classList.add('hidden');
    document.getElementById('register-form').classList.add('hidden');

    if (type === 'login') {
        document.getElementById('login-form').classList.remove('hidden');
        event.target.classList.add('active'); // Warning: depends on click context
    } else {
        document.getElementById('register-form').classList.remove('hidden');
        // Hack for tab highlighting if called from buttons
        document.querySelectorAll('.tab')[1].classList.add('active');
    }
}

async function handleLogin(e) {
    e.preventDefault();
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;

    // FormData for OAuth2 standard
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    try {
        const res = await fetch(`${API_URL}/token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData
        });

        if (!res.ok) throw new Error("Invalid credentials");

        const data = await res.json();
        token = data.access_token;
        localStorage.setItem("access_token", token);
        showDashboard();
        fetchUser();
    } catch (err) {
        document.getElementById('auth-msg').innerText = err.message;
    }
}

async function handleRegister(e) {
    e.preventDefault();
    const username = document.getElementById('reg-username').value;
    const password = document.getElementById('reg-password').value;

    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    try {
        const res = await fetch(`${API_URL}/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData
        });

        if (!res.ok) throw new Error("Registration failed");

        alert("Account created! Please login.");
        showAuthForm('login');
        document.querySelectorAll('.tab')[0].click(); // Switch tab
    } catch (err) {
        document.getElementById('auth-msg').innerText = err.message;
    }
}

function logout() {
    token = null;
    localStorage.removeItem("access_token");
    location.reload();
}

// Dashboard Functions
function showDashboard() {
    document.getElementById('auth-view').classList.add('hidden');
    document.getElementById('dashboard-view').classList.remove('hidden');
}

async function fetchUser() {
    try {
        const res = await fetch(`${API_URL}/users/me`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await res.json();
        document.getElementById('user-display').innerText = data.username;
    } catch (e) {
        logout();
    }
}

async function getRecommendation() {
    const mood = document.getElementById('mood-input').value;
    if (!mood) return;

    const btn = document.getElementById('rec-btn');
    btn.innerText = "Thinking... 🧠";
    btn.disabled = true;

    try {
        const res = await fetch(`${API_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ mood: mood })
        });

        const data = await res.json();

        // Update UI
        const container = document.getElementById('results-container');
        container.innerHTML = ''; // Clear previous
        container.classList.remove('hidden');

        const template = document.getElementById('card-template');

        data.forEach(movie => {
            const clone = template.content.cloneNode(true);
            clone.querySelector('.movie-title').innerText = movie.title;
            clone.querySelector('.release-date').innerText = movie.release_date;
            clone.querySelector('.genres-list').innerText = movie.genres;
            clone.querySelector('.score-val').innerText = movie.genre_match_score.toFixed(2);


            // Bind movie ID to buttons
            const buttons = clone.querySelectorAll('.stars button');
            buttons.forEach(btn => {
                btn.onclick = () => rateMovie(movie.movie_id, parseInt(btn.innerText === '⭐' ? 1 : 1)); // Fix logic usage
                // Actually my template logic for onclick="rateMovie(this, 1)" passed element. 
                // Let's rewrite strictly in JS for clarity
            });

            // Re-bind properly
            clone.querySelectorAll('.stars button').forEach((btn, index) => {
                btn.onclick = () => rateMovie(movie.movie_id, index + 1, btn);
            });

            container.appendChild(clone);
        });

    } catch (e) {
        console.error(e);
        alert("Error getting recommendation");
    } finally {
        btn.innerText = "Get Recommendation ✨";
        btn.disabled = false;
    }
}

async function rateMovie(movieId, rating, btnElement) {
    try {
        await fetch(`${API_URL}/rate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ movie_id: movieId, rating: rating })
        });

        // Visual Feedback - Shine stars
        if (btnElement) {
            const container = btnElement.parentElement;
            const stars = container.querySelectorAll('button');
            stars.forEach((star, index) => {
                if (index < rating) {
                    star.classList.add('star-active');
                    star.innerText = '★';
                } else {
                    star.classList.remove('star-active');
                    star.innerText = '★';
                }
            });

            // Optional: Disable further voting?
            // container.style.pointerEvents = 'none'; 
        }

    } catch (e) {
        alert("Failed to save rating");
    }
}
