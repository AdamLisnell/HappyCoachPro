import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'
import { VitePWA } from 'vite-plugin-pwa'
import type { Plugin } from 'vite'

// Local dev handler for /api/coaching — mirrors the Vercel edge function logic
function localApiPlugin(apiKey: string | undefined): Plugin {
  return {
    name: 'local-api',
    configureServer(server) {
      server.middlewares.use('/api/coaching', async (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405; res.end('Method not allowed'); return
        }
        if (!apiKey) {
          res.setHeader('Content-Type', 'application/json')
          res.statusCode = 503
          res.end(JSON.stringify({ error: 'Create frontend/.env with ANTHROPIC_API_KEY=sk-ant-... to enable coaching locally' }))
          return
        }
        let body = ''
        req.on('data', (c: Buffer) => { body += c.toString() })
        req.on('end', async () => {
          try {
            const parsed = JSON.parse(body)
            const { analysis, cameraAngle = 'side' } = parsed

            const clubCtx = (c: string) =>
              (c === 'driver' || c === 'wood_3' || c === 'wood_5') ? 'Driver/Wood: wider arc, upward angle of attack, shoulder turn ≥90°.'
              : c === 'putter' ? 'Putter: pendulum shoulder motion only.'
              : (c.startsWith('iron_') || c === 'hybrid') ? 'Mid/short iron: ball-first with small divot, steeper plane, 80–90° turn.'
              : 'Wedge: compact controlled swing, hands well ahead at impact.'

            // Build user message matching the edge function
            const lines: string[] = [
              `Camera: ${cameraAngle === 'behind' ? 'Down-the-line' : 'Face-on / side-on'}`,
              `Club: ${analysis.club} — ${clubCtx(analysis.club)}`,
              `Overall: ${analysis.overall_score}/100`,
            ]
            if (analysis.tempo_ratio !== undefined) lines.push(`Tempo ratio (BS:DS): ${analysis.tempo_ratio.toFixed(2)}:1`)
            if (analysis.x_factor_top !== undefined) lines.push(`X-factor at top: ${analysis.x_factor_top.toFixed(0)}°`)
            lines.push('', 'Subscores:')
            const fmt = (s: { score: number; grade: string; feedback: string; details?: string | null }) =>
              `  ${s.score}/100 (${s.grade}): ${s.feedback}${s.details ? ' — ' + s.details : ''}`
            if (analysis.posture_score)  lines.push('  Posture'  + fmt(analysis.posture_score))
            if (analysis.tempo_score)    lines.push('  Tempo'    + fmt(analysis.tempo_score))
            if (analysis.rotation_score) lines.push('  Rotation' + fmt(analysis.rotation_score))
            if (analysis.balance_score)  lines.push('  Balance'  + fmt(analysis.balance_score))
            if (analysis.phases?.length) {
              lines.push('', 'Phase angles:')
              for (const p of analysis.phases) {
                const angles = Object.entries(p.angles).filter(([,v]) => v !== null)
                  .map(([k,v]) => `${k}=${(v as number).toFixed(1)}°`).join(', ')
                if (angles) lines.push(`  ${p.phase} (score ${p.score}/100): ${angles}`)
              }
            }
            lines.push('', 'OPTIMAL: Top shoulder 80–100°, hip 35–50°, X-factor ≥45°. Impact spine 25–40°, hip 35–50°. Tempo 3:1.')

            const SYSTEM = `You are an elite PGA-level golf instructor. Respond with a valid JSON object with EXACTLY these four fields:
"narrative": 3 sentences. Start with the biggest flaw and its measured value, then what's working. No filler.
"ball_flight": 1 sentence explaining the likely ball-flight outcome (slice/push/fat/thin/low-launch) and why.
"focus_areas": array of 3 strings: "[Joint]: [measured]° vs ideal [range]. [Verdict]."
"practice_plan": 4-6 sentences with EXACTLY 2 named drills (reps + feel cues) and one end-of-session feel checkpoint.
Tailor to club type. No filler phrases. JSON only — no markdown.`

            const apiRes = await fetch('https://api.anthropic.com/v1/messages', {
              method: 'POST',
              headers: {
                'x-api-key': apiKey,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json',
              },
              body: JSON.stringify({
                model: 'claude-sonnet-4-6',
                max_tokens: 1100,
                system: SYSTEM,
                messages: [{ role: 'user', content: lines.join('\n') }],
              }),
            })
            const apiData = await apiRes.json() as { content?: Array<{ type: string; text: string }> }
            const raw = apiData.content?.[0]?.type === 'text' ? apiData.content[0].text : '{}'
            const text = raw.replace(/^```(?:json)?\s*/m, '').replace(/\s*```\s*$/m, '').trim()
            const result = JSON.parse(text)
            res.setHeader('Content-Type', 'application/json')
            res.statusCode = 200
            res.end(JSON.stringify({ ...result, generated_at: new Date().toISOString() }))
          } catch (err) {
            console.error('[local-api]', err)
            res.setHeader('Content-Type', 'application/json')
            res.statusCode = 500
            res.end(JSON.stringify({ error: 'Local coaching handler failed' }))
          }
        })
      })
    },
  }
}

export default defineConfig(({ mode }) => {
  // Load .env so the server-side plugin can access ANTHROPIC_API_KEY
  const env = loadEnv(mode, process.cwd(), '')

  return defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    localApiPlugin(env.ANTHROPIC_API_KEY),
    VitePWA({
      registerType: 'autoUpdate',
      manifest: {
        name: 'HappyCoachPro',
        short_name: 'HappyCoach',
        description: 'AI-powered golf swing analyzer',
        theme_color: '#0A1F14',
        background_color: '#0A1F14',
        display: 'standalone',
        orientation: 'portrait',
        start_url: '/',
        icons: [
          { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        runtimeCaching: [
          { urlPattern: /^\/api\//, handler: 'NetworkOnly' },
          {
            urlPattern: /^https:\/\/storage\.googleapis\.com\/mediapipe-models\//,
            handler: 'CacheFirst',
            options: { cacheName: 'mediapipe-models', expiration: { maxAgeSeconds: 60 * 60 * 24 * 30 } },
          },
          {
            urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/npm\/@mediapipe\//,
            handler: 'CacheFirst',
            options: { cacheName: 'mediapipe-wasm', expiration: { maxAgeSeconds: 60 * 60 * 24 * 30 } },
          },
        ],
      },
    }),
  ],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 3000,
    host: true,
  },
  })
})
