// Gestionnaire des questions avec chargement asynchrone
const QuestionsManager = {
	data: null,
	loaded: false,

	// Charger les données depuis le fichier JSON
	async loadQuestions() {
		try {
			const response = await fetch('questions.json');
			if (!response.ok) {
				throw new Error(`Erreur HTTP: ${response.status}`);
			}
			this.data = await response.json();
			this.loaded = true;
			console.log('Questions chargées avec succès');
			return this.data;
		} catch (error) {
			console.error('Erreur lors du chargement des questions:', error);
			// Retourner des données par défaut en cas d'erreur
			return this.getDefaultData();
		}
	},

	// Données par défaut en cas d'échec du chargement
	getDefaultData() {
		return {
			categories: [{ id: 'default', name: 'Questions' }],
			questions: {
				default: [{
					id: 1,
					text: "Question par défaut - Le fichier questions.json n'a pas pu être chargé.",
					type: "qcm",
					answers: ["Option 1", "Option 2", "Option 3"],
					correctAnswer: "Option 1",
					explanation: "Vérifiez que le fichier data/questions.json existe.",
					figure: null
				}]
			}
		};
	},

	// Obtenir toutes les catégories
	getCategories() {
		if (!this.loaded) {
			console.warn('Les questions ne sont pas encore chargées');
			return [];
		}
		return this.data.categories;
	},

	// Obtenir les questions d'une catégorie
	getQuestions(categoryId) {
		if (!this.loaded) {
			console.warn('Les questions ne sont pas encore chargées');
			return [];
		}
		return this.data.questions[categoryId] || [];
	},

	// Obtenir toutes les questions (pour les statistiques)
	getAllQuestions() {
		if (!this.loaded) {
			console.warn('Les questions ne sont pas encore chargées');
			return [];
		}

		const allQuestions = [];
		Object.values(this.data.questions).forEach(categoryQuestions => {
			allQuestions.push(...categoryQuestions);
		});
		return allQuestions;
	},

	// Compter le nombre total de questions
	getTotalQuestionCount() {
		if (!this.loaded) return 0;

		let count = 0;
		Object.values(this.data.questions).forEach(categoryQuestions => {
			count += categoryQuestions.length;
		});
		return count;
	},

	// Obtenir les questions qui utilisent Pythagore
	getPythagorasQuestions() {
		if (!this.loaded) return [];

		const pythagorasQuestions = [];
		Object.entries(this.data.questions).forEach(([categoryId, categoryQuestions]) => {
			categoryQuestions.forEach(question => {
				if (question.usesPythagoras) {
					const category = this.data.categories.find(cat => cat.id === categoryId);
					pythagorasQuestions.push({
						category: category ? category.name : 'Inconnu',
						text: question.text,
						explanation: question.explanation,
						categoryId: categoryId,
						questionId: question.id
					});
				}
			});
		});
		return pythagorasQuestions;
	},

	// Trouver une question spécifique
	findQuestion(categoryId, questionId) {
		if (!this.loaded) return null;

		const categoryQuestions = this.data.questions[categoryId];
		if (!categoryQuestions) return null;

		return categoryQuestions.find(q => q.id === questionId) || null;
	},

	// Vérifier si une catégorie existe
	categoryExists(categoryId) {
		if (!this.loaded) return false;
		return this.data.categories.some(cat => cat.id === categoryId);
	}
};

// Initialiser le chargement des questions
QuestionsManager.loadQuestions();