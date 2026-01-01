const { createApp, ref, computed, onMounted, watch } = Vue;

createApp({
	setup() {
		// États de l'application
		const categories = ref([]);
		const questions = ref({});
		const currentCategory = ref('');
		const currentQuestionIndex = ref(0);
		const selectedAnswer = ref(null);
		const numericAnswer = ref(null);
		const answerSubmitted = ref(false);
		const answerIsCorrect = ref(false);
		const correctAnswers = ref(0);
		const totalQuestions = ref(0);
		const totalScore = ref(0);
		const totalQuestionsAttempted = ref(0);
		const pythagorasQuestions = ref([]);
		const isLoading = ref(true);
		const figureCanvas = ref(null);

		// Initialiser l'application
		const initApp = async () => {
			isLoading.value = true;
			try {
				// Charger les questions
				const questionsData = await QuestionsManager.loadQuestions();
				categories.value = QuestionsManager.getCategories();
				questions.value = questionsData.questions;

				// Initialiser avec la première catégorie
				if (categories.value.length > 0) {
					currentCategory.value = categories.value[0].id;
					totalQuestions.value = QuestionsManager.getQuestions(currentCategory.value).length;
					pythagorasQuestions.value = QuestionsManager.getPythagorasQuestions();
				}

				console.log(`Application initialisée avec ${categories.value.length} catégories`);
				console.log(`${pythagorasQuestions.value.length} questions utilisent Pythagore`);
			} catch (error) {
				console.error('Erreur lors de l\'initialisation:', error);
			} finally {
				isLoading.value = false;
			}
		};

		// Question actuelle
		const currentQuestion = computed(() => {
			const categoryQuestions = QuestionsManager.getQuestions(currentCategory.value);
			return categoryQuestions ? categoryQuestions[currentQuestionIndex.value] : {};
		});

		// Progression
		const progress = computed(() => {
			const categoryQuestions = QuestionsManager.getQuestions(currentCategory.value);
			return categoryQuestions ? ((currentQuestionIndex.value + 1) / categoryQuestions.length) * 100 : 0;
		});

		// Score global
		const overallScore = computed(() => {
			if (totalQuestionsAttempted.value === 0) return 0;
			return Math.round((totalScore.value / totalQuestionsAttempted.value) * 100);
		});

		// Sélectionner une catégorie
		const selectCategory = (categoryId) => {
			if (!QuestionsManager.categoryExists(categoryId)) {
				console.error('Catégorie non trouvée:', categoryId);
				return;
			}

			currentCategory.value = categoryId;
			currentQuestionIndex.value = 0;
			resetQuestionState();
			totalQuestions.value = QuestionsManager.getQuestions(categoryId).length;
			drawFigure();
		};

		// ... [le reste des fonctions reste identique] ...

		// Afficher les questions Pythagore
		const showPythagorasQuestions = () => {
			const count = pythagorasQuestions.value.length;
			let message = `Il y a ${count} question${count > 1 ? 's' : ''} utilisant le théorème de Pythagore:\n\n`;

			pythagorasQuestions.value.forEach((q, index) => {
				message += `${index + 1}. [${q.category}] ${q.text}\n`;
			});

			alert(message);
		};

		// Initialiser au chargement
		onMounted(() => {
			initApp();
		});

		return {
			categories,
			currentCategory,
			currentQuestion,
			currentQuestionIndex,
			selectedAnswer,
			numericAnswer,
			answerSubmitted,
			answerIsCorrect,
			correctAnswers,
			totalQuestions,
			totalScore,
			totalQuestionsAttempted,
			overallScore,
			pythagorasQuestions,
			isLoading,
			figureCanvas,
			progress,
			selectCategory,
			selectAnswer,
			submitAnswer,
			nextQuestion,
			formatCorrectAnswer,
			showPythagorasQuestions,
			drawFigure
		};
	}
}).mount('#app');