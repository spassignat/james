<template>
	<div class="chat-container">
		<!-- En-tête du chat -->
		<el-header class="chat-header">
			<h2>Chat avec Ollama</h2>
			<el-select
					v-model="selectedModel"
					placeholder="Sélectionner un modèle"
					@change="loadChatHistory"
					class="model-selector"
			>
				<el-option
						v-for="model in availableModels"
						:key="model"
						:label="model"
						:value="model"
				/>
			</el-select>
		</el-header>

		<!-- Zone de messages avec infinite scroll -->
		<el-main class="messages-container" ref="messagesContainer">
			<div
					v-infinite-scroll="loadMoreMessages"
					:infinite-scroll-disabled="loading || !hasMoreMessages"
					infinite-scroll-distance="100"
					class="messages-wrapper"
			>
				<!-- Indicateur de chargement pour les anciens messages -->
				<div v-if="loadingMore" class="loading-indicator">
					<el-icon class="is-loading">
						<Loading />
					</el-icon>
					Chargement des messages...
				</div>

				<!-- Messages -->
				<div
						v-for="(message, index) in messages"
						:key="index"
						:class="['message', message.role]"
				>
					<!-- Avatar -->
					<div class="avatar">
						<el-avatar :style="message.role === 'user' ?
              { backgroundColor: '#409EFF' } :
              { backgroundColor: '#67C23A' }">
							{{ message.role === 'user' ? 'U' : 'AI' }}
						</el-avatar>
					</div>

					<!-- Contenu du message -->
					<div class="message-content">
						<div class="message-header">
							<strong>{{ message.role === 'user' ? 'Vous' : 'Ollama' }}</strong>
							<span class="timestamp">{{ formatTime(message.timestamp) }}</span>
						</div>

						<!-- Message textuel -->
						<div
								v-if="!message.isCode"
								class="message-text"
								v-html="formatMessage(message.content)"
						></div>

						<!-- Bloc de code avec bouton copier -->
						<div v-else class="code-block">
							<div class="code-header">
								<span class="language-badge">{{ message.language || 'code' }}</span>
								<el-button
										size="small"
										type="primary"
										:icon="DocumentCopy"
										@click="copyToClipboard(message.content)"
										class="copy-button"
								>
									Copier
								</el-button>
							</div>
							<pre><code>{{ message.content }}</code></pre>
						</div>

						<!-- Indicateur de streaming -->
						<div v-if="message.streaming" class="streaming-indicator">
							<el-icon class="is-loading">
								<Loading />
							</el-icon>
						</div>
					</div>
				</div>

				<!-- Indicateur de chargement initial -->
				<div v-if="loading" class="loading-full">
					<el-icon class="is-loading">
						<Loading />
					</el-icon>
					Chargement de l'historique...
				</div>

				<!-- Aucun message -->
				<div v-if="!loading && messages.length === 0" class="empty-state">
					<el-empty description="Aucun message. Commencez une conversation !" />
				</div>
			</div>
		</el-main>

		<!-- Zone de saisie -->
		<el-footer class="input-container">
			<div class="input-wrapper">
				<el-input
						v-model="inputMessage"
						type="textarea"
						:autosize="{ minRows: 3, maxRows: 1000 }"
						placeholder="Tapez votre message..."
						@keydown.enter.exact.prevent="sendMessage"
						:disabled="streaming"
						resize="none"
						class="message-input"
				/>
				<el-button
						type="primary"
						:icon="Promotion"
						@click="sendMessage"
						:loading="streaming"
						:disabled="!inputMessage.trim() || streaming"
						class="send-button"
				>
					Envoyer
				</el-button>
			</div>

			<!-- Options supplémentaires -->
			<div class="input-options">
				<el-checkbox v-model="autoScroll" label="Défilement automatique" />
				<el-checkbox v-model="streamResponse" label="Réponse en streaming" />
				<el-tooltip content="Nouvelle conversation">
					<el-button
							:icon="Refresh"
							@click="newChat"
							circle
					/>
				</el-tooltip>
			</div>
		</el-footer>
	</div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch, computed } from 'vue'
import axios from 'axios'
import {
	ElMessage,
	ElMessageBox,
	ElLoading
} from 'element-plus'
import {
	Promotion,
	DocumentCopy,
	Refresh,
	Loading
} from '@element-plus/icons-vue'

// Configuration Ollama
const OLLAMA_BASE_URL = 'http://localhost:11434'
const MESSAGES_PER_PAGE = 20

// Références
const messagesContainer = ref(null)
const inputMessage = ref('')
const messages = ref([])
const streaming = ref(false)
const loading = ref(false)
const loadingMore = ref(false)
const hasMoreMessages = ref(true)
const page = ref(1)
const autoScroll = ref(true)
const streamResponse = ref(true)
const selectedModel = ref('llama2')
const availableModels = ref([])

// Détecter les blocs de code dans le texte
const detectCodeBlocks = (text) => {
	const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g
	const parts = []
	let lastIndex = 0
	let match

	while ((match = codeBlockRegex.exec(text)) !== null) {
		// Texte avant le bloc de code
		if (match.index > lastIndex) {
			parts.push({
				type: 'text',
				content: text.slice(lastIndex, match.index)
			})
		}

		// Bloc de code
		parts.push({
			type: 'code',
			language: match[1] || '',
			content: match[2].trim()
		})

		lastIndex = match.index + match[0].length
	}

	// Texte après le dernier bloc de code
	if (lastIndex < text.length) {
		parts.push({
			type: 'text',
			content: text.slice(lastIndex)
		})
	}

	// Si aucun bloc de code n'a été trouvé, retourner tout comme texte
	if (parts.length === 0) {
		parts.push({
			type: 'text',
			content: text
		})
	}

	return parts
}

// Formater un message pour l'affichage
const formatMessage = (content) => {
	const parts = detectCodeBlocks(content)
	return parts.map(part => {
		if (part.type === 'code') {
			return `<div class="inline-code-block">
                <span class="inline-language">${part.language || 'code'}</span>
                <pre><code>${escapeHtml(part.content)}</code></pre>
              </div>`
		} else {
			return escapeHtml(part.content).replace(/\n/g, '<br>')
		}
	}).join('')
}

// Échapper le HTML
const escapeHtml = (text) => {
	const div = document.createElement('div')
	div.textContent = text
	return div.innerHTML
}

// Charger les modèles disponibles
const loadModels = async () => {
	try {
		const response = await axios.get(`${OLLAMA_BASE_URL}/api/tags`)
		availableModels.value = response.data.models.map(model => model.name)
	} catch (error) {
		console.error('Erreur lors du chargement des modèles:', error)
		ElMessage.error('Impossible de charger les modèles Ollama')
	}
}

// Charger l'historique des messages
const loadChatHistory = async () => {
	loading.value = true
	try {
		// Dans un vrai scénario, vous auriez une API backend pour l'historique
		// Pour l'exemple, on initialise avec un message vide
		messages.value = []
		page.value = 1
		hasMoreMessages.value = true
	} catch (error) {
		console.error('Erreur lors du chargement de l\'historique:', error)
		ElMessage.error('Impossible de charger l\'historique')
	} finally {
		loading.value = false
	}
}

// Charger plus de messages (infinite scroll)
const loadMoreMessages = async () => {
	if (loadingMore.value || !hasMoreMessages.value) return

	loadingMore.value = true
	try {
		// Simuler le chargement depuis une API
		await new Promise(resolve => setTimeout(resolve, 1000))

		// Dans un vrai scénario, vous appelleriez votre API avec pagination
		// messages.value = [...oldMessages, ...newMessages]

		hasMoreMessages.value = messages.value.length < 100 // Exemple: arrêter après 100 messages
		page.value += 1
	} catch (error) {
		console.error('Erreur lors du chargement des messages:', error)
	} finally {
		loadingMore.value = false
	}
}

// Envoyer un message
const sendMessage = async () => {
	const message = inputMessage.value.trim()
	if (!message || streaming.value) return

	// Ajouter le message de l'utilisateur
	const userMessage = {
		role: 'user',
		content: message,
		timestamp: new Date(),
		isCode: false
	}

	messages.value.push(userMessage)
	inputMessage.value = ''

	// Ajouter un message d'assistant vide pour le streaming
	const assistantMessage = {
		role: 'assistant',
		content: '',
		timestamp: new Date(),
		streaming: true,
		isCode: false
	}

	messages.value.push(assistantMessage)
	streaming.value = true

	// Faire défiler vers le bas
	if (autoScroll.value) {
		await nextTick()
		scrollToBottom()
	}

	try {
		if (streamResponse.value) {
			await streamOllamaResponse(message, assistantMessage)
		} else {
			await getOllamaResponse(message, assistantMessage)
		}
	} catch (error) {
		console.error('Erreur:', error)
		ElMessage.error('Erreur lors de la communication avec Ollama')
		// Supprimer le message d'assistant en cas d'erreur
		messages.value.pop()
	} finally {
		streaming.value = false
		assistantMessage.streaming = false
	}
}

// Obtenir une réponse streaming d'Ollama
const streamOllamaResponse = async (prompt, assistantMessage) => {
	try {
		const response = await fetch(`${OLLAMA_BASE_URL}/api/generate`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				model: selectedModel.value,
				prompt: prompt,
				stream: true
			})
		})

		const reader = response.body.getReader()
		const decoder = new TextDecoder()
		let fullResponse = ''

		while (true) {
			const { done, value } = await reader.read()
			if (done) break

			const chunk = decoder.decode(value)
			const lines = chunk.split('\n').filter(line => line.trim())

			for (const line of lines) {
				try {
					const parsed = JSON.parse(line)
					if (parsed.response) {
						fullResponse += parsed.response
						assistantMessage.content = fullResponse

						// Détecter si c'est du code
						const codeMatch = fullResponse.match(/```(\w+)?\n([\s\S]*?)```/)
						if (codeMatch) {
							assistantMessage.isCode = true
							assistantMessage.language = codeMatch[1] || ''
						}

						// Mettre à jour l'interface
						await nextTick()

						// Faire défiler vers le bas pendant le streaming
						if (autoScroll.value) {
							scrollToBottom()
						}
					}
				} catch (e) {
					// Ignorer les erreurs de parsing JSON
				}
			}
		}
	} catch (error) {
		throw error
	}
}

// Obtenir une réponse complète d'Ollama (non streaming)
const getOllamaResponse = async (prompt, assistantMessage) => {
	try {
		const response = await axios.post(`${OLLAMA_BASE_URL}/api/chat`, {
			model: selectedModel.value,
			prompt: prompt,
			stream: false
		})

		assistantMessage.content = response.data.response
		assistantMessage.streaming = false

		// Détecter si c'est du code
		const codeMatch = response.data.response.match(/```(\w+)?\n([\s\S]*?)```/)
		if (codeMatch) {
			assistantMessage.isCode = true
			assistantMessage.language = codeMatch[1] || ''
		}

		// Faire défiler vers le bas
		if (autoScroll.value) {
			await nextTick()
			scrollToBottom()
		}
	} catch (error) {
		throw error
	}
}

// Copier dans le presse-papier
const copyToClipboard = async (text) => {
	try {
		await navigator.clipboard.writeText(text)
		ElMessage.success('Copié dans le presse-papier !')
	} catch (err) {
		console.error('Erreur lors de la copie:', err)
		ElMessage.error('Échec de la copie')
	}
}

// Faire défiler vers le bas
const scrollToBottom = () => {
	if (messagesContainer.value) {
		const container = messagesContainer.value.$el || messagesContainer.value
		container.scrollTop = container.scrollHeight
	}
}

// Formater l'heure
const formatTime = (date) => {
	return new Date(date).toLocaleTimeString('fr-FR', {
		hour: '2-digit',
		minute: '2-digit'
	})
}

// Nouvelle conversation
const newChat = async () => {
	try {
		await ElMessageBox.confirm(
			'Voulez-vous vraiment commencer une nouvelle conversation ?',
			'Confirmation',
			{
				confirmButtonText: 'Oui',
				cancelButtonText: 'Non',
				type: 'warning'
			}
		)

		messages.value = []
		page.value = 1
		hasMoreMessages.value = true
		ElMessage.success('Nouvelle conversation commencée')
	} catch {
		// Annulé
	}
}

// Initialisation
onMounted(() => {
	loadModels()
	loadChatHistory()
})

// Watchers
watch(() => messages.value.length, () => {
	if (autoScroll.value && !loadingMore.value) {
		nextTick(() => scrollToBottom())
	}
})
</script>

<style scoped>
.chat-container {
	height: 90vh;
	display: flex;
	flex-direction: column;
	background-color: #f5f7fa;
}

.chat-header {
	display: flex;
	justify-content: space-between;
	align-items: center;
	background-color: white;
	border-bottom: 1px solid #e4e7ed;
	box-shadow: 0 2px 4px rgba(0,0,0,0.1);
	z-index: 100;
}

.model-selector {
	width: 200px;
}

.messages-container {
	flex: 1;
	overflow-y: auto;
	padding: 20px;
	background-color: #f5f7fa;
}

.messages-wrapper {
	max-width: 800px;
	margin: 0 auto;
	display: flex;
	flex-direction: column;
	gap: 20px;
}

.message {
	display: flex;
	gap: 12px;
	animation: fadeIn 0.3s ease;
}

.message.user {
	flex-direction: row-reverse;
}

.message-content {
	max-width: 70%;
	background-color: white;
	border-radius: 12px;
	padding: 12px 16px;
	box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.message.user .message-content {
	background-color: #409EFF;
	color: white;
}

.message-header {
	display: flex;
	justify-content: space-between;
	align-items: center;
	margin-bottom: 8px;
	font-size: 0.9em;
}

.timestamp {
	font-size: 0.8em;
	opacity: 0.7;
}

.message.user .timestamp {
	color: rgba(255,255,255,0.8);
}

.message-text {
	line-height: 1.5;
	white-space: pre-wrap;
	word-wrap: break-word;
}

.code-block {
	background-color: #1e1e1e;
	border-radius: 8px;
	overflow: hidden;
	margin: 8px 0;
}

.code-header {
	display: flex;
	justify-content: space-between;
	align-items: center;
	padding: 8px 12px;
	background-color: #2d2d2d;
	color: #ccc;
}

.language-badge {
	font-size: 0.8em;
	padding: 2px 8px;
	background-color: #404040;
	border-radius: 4px;
}

.code-block pre {
	margin: 0;
	padding: 12px;
	overflow-x: auto;
}

.code-block code {
	color: #d4d4d4;
	font-family: 'Consolas', 'Monaco', monospace;
	font-size: 0.9em;
}

.copy-button {
	padding: 4px 8px;
}

.inline-code-block {
	background-color: #1e1e1e;
	border-radius: 6px;
	padding: 10px;
	margin: 8px 0;
	overflow-x: auto;
}

.inline-language {
	display: inline-block;
	background-color: #404040;
	color: #ccc;
	padding: 2px 8px;
	border-radius: 4px;
	font-size: 0.8em;
	margin-bottom: 8px;
}

.input-container {
	padding: 20px;
	background-color: white;
	border-top: 1px solid #e4e7ed;
	box-shadow: 0 -2px 4px rgba(0,0,0,0.1);
}

.input-wrapper {
	display: flex;
	gap: 12px;
	margin-bottom: 12px;
}

.message-input {
	flex: 1;
}

.send-button {
	align-self: flex-end;
	height: fit-content;
}

.input-options {
	display: flex;
	justify-content: space-between;
	align-items: center;
}

.loading-indicator,
.loading-full {
	display: flex;
	justify-content: center;
	align-items: center;
	gap: 8px;
	padding: 20px;
	color: #909399;
}

.streaming-indicator {
	display: flex;
	justify-content: center;
	padding-top: 8px;
}

.empty-state {
	display: flex;
	justify-content: center;
	align-items: center;
	height: 200px;
}

@keyframes fadeIn {
	from {
		opacity: 0;
		transform: translateY(10px);
	}
	to {
		opacity: 1;
		transform: translateY(0);
	}
}

/* Responsive */
@media (max-width: 768px) {
	.chat-header {
		flex-direction: column;
		gap: 10px;
		padding: 10px;
	}

	.model-selector {
		width: 100%;
	}

	.message-content {
		max-width: 85%;
	}

	.input-wrapper {
		flex-direction: column;
	}

	.send-button {
		align-self: stretch;
	}
}
</style>