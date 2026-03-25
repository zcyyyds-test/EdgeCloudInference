const BASE = ''

export interface VisionOutput {
  timestamp: number
  anomaly_level: number
  measurement_confidence: number
  color_rgb: [number, number, number]
  secondary_metric: number
  texture_irregularity: number
  surface_uniformity: number
  anomaly_score: number
  anomaly_confidence: number
  inference_latency_ms: number
  image_path?: string
}

export interface AnalysisResult {
  judgment: 'normal' | 'warning' | 'alarm'
  confidence: number
  suggested_action: string
  reasoning: string
  root_cause: string
  latency_ms: number
  source: 'edge' | 'cloud'
}

export interface DetectResponse {
  vision_output: VisionOutput
  scenario_name: string
  scenario_difficulty: string
}

export interface HealthStatus {
  status: string
  edge_available: boolean
  cloud_available: boolean
}

export interface ExperimentMeta {
  name: string
  file: string
  size_kb: number
}

export interface TierResult {
  tier: string
  triggered: boolean
  detail: string
}

export interface RouteResponse {
  tier: string
  reason: string
  action: string
  latency_ms: number
  tier_index: number
  tier_name: string
  tiers: TierResult[]
}

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  health: () => fetchJSON<HealthStatus>('/health'),

  detect: (scenario_key?: string, difficulty?: string) =>
    fetchJSON<DetectResponse>('/detect', {
      method: 'POST',
      body: JSON.stringify({ scenario_key, difficulty }),
    }),

  analyzeEdge: (vision_output: VisionOutput) =>
    fetchJSON<AnalysisResult>('/analyze/edge', {
      method: 'POST',
      body: JSON.stringify({ vision_output, source: 'edge' }),
    }),

  analyzeCloud: (vision_output: VisionOutput) =>
    fetchJSON<AnalysisResult>('/analyze/cloud', {
      method: 'POST',
      body: JSON.stringify({ vision_output, source: 'cloud' }),
    }),

  route: (vision_output: VisionOutput, num_correlated_anomalies = 0) =>
    fetchJSON<RouteResponse>('/api/demo/route', {
      method: 'POST',
      body: JSON.stringify({ vision_output, num_correlated_anomalies }),
    }),

  experiments: () => fetchJSON<ExperimentMeta[]>('/api/experiments'),
  experiment: (name: string) => fetchJSON<Record<string, unknown>>(`/api/experiments/${name}`),
  config: () => fetchJSON<Record<string, unknown>>('/api/system/config'),
  models: () => fetchJSON<Record<string, unknown>>('/api/system/models'),
}
