// main.js
import { createApp } from 'vue'
import { createPinia } from 'pinia' // Optionnel: pour le state management
import App from './App.vue'
import router from './router' // Optionnel: pour le routing

// Importation d'Element Plus
import ElementPlus, {ElLoading, ElMessageBox} from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

// Importation des styles globaux
import '/assets/styles/main.css'

// Création de l'application
const app = createApp(App)

// Installation d'Element Plus
app.use(ElementPlus)

// Installation des icônes
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
	app.component(key, component)
}

// Installation de Pinia (optionnel)
const pinia = createPinia()
app.use(pinia)

// Installation du router (optionnel)
app.use(router)

// Montage de l'application
app.mount('#app')

// Configuration globale des messages
import { ElMessage } from 'element-plus'

app.config.globalProperties.$message = ElMessage
app.config.globalProperties.$messageBox = ElMessageBox
app.config.globalProperties.$loading = ElLoading