import { StrictMode } from 'react'                     /* helps identify potential problems */
import { createRoot } from 'react-dom/client'          /* mount a React app */
import './index.css'                                   /* Loads global styles (Tailwind or custom CSS) */
import App from './App.jsx'                            /* imports app.jsx */

createRoot(document.getElementById('root')).render(           /* eenders the entire React app into the HTML root */
  <StrictMode>
    <App />                         
  </StrictMode>,
)
