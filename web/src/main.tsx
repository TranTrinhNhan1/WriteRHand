import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/geist-mono/wght.css'
import '@fontsource-variable/geist/wght.css'
import './styles/global.css'
import './styles/recognise.css'
import './styles/explorer-shared.css'
import './styles/explorer-classic.css'
import './styles/explorer.css'
import App from './App'
import { LangProvider } from './i18n'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LangProvider>
      <App />
    </LangProvider>
  </StrictMode>,
)
