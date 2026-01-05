// axios-config.js
import axios from 'axios'

// Configuration globale d'Axios
const ollamaApi = axios.create({
	baseURL: 'http://localhost:11434/api',
	timeout: 30000, // 30 secondes
	headers: {
		'Content-Type': 'application/json'
	}
})

// Intercepteur pour les requêtes
ollamaApi.interceptors.request.use(
	config => {
		// Vous pouvez ajouter des headers d'authentification ici si nécessaire
		console.log('Requête envoyée:', config.method, config.url)
		return config
	},
	error => {
		console.error('Erreur de requête:', error)
		return Promise.reject(error)
	}
)

// Intercepteur pour les réponses
ollamaApi.interceptors.response.use(
	response => {
		console.log('Réponse reçue:', response.status, response.config.url)
		return response
	},
	error => {
		console.error('Erreur de réponse:', error)

		// Gestion des erreurs spécifiques
		if (error.code === 'ECONNREFUSED') {
			console.error('Ollama n\'est pas accessible. Assurez-vous qu\'il est en cours d\'exécution.')
		} else if (error.response) {
			switch (error.response.status) {
				case 404:
					console.error('Modèle non trouvé')
					break
				case 500:
					console.error('Erreur interne du serveur Ollama')
					break
			}
		}

		return Promise.reject(error)
	}
)

export { ollamaApi }