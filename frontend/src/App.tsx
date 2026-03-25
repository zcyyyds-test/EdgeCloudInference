import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Overview from './pages/Overview'
import LiveDemo from './pages/LiveDemo'
import ModelAblation from './pages/ModelAblation'
import Experiments from './pages/Experiments'
import Architecture from './pages/Architecture'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/demo" element={<LiveDemo />} />
        <Route path="/ablation" element={<ModelAblation />} />
        <Route path="/experiments" element={<Experiments />} />
        <Route path="/architecture" element={<Architecture />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}

export default App
