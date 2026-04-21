import Anthropic from '@anthropic-ai/sdk';
import type { SwingAnalysis } from '../src/types';

const SYSTEM_PROMPT = `You are an elite PGA-level golf instructor. You receive biomechanical swing data — measured angles, scores, and phase breakdowns — and deliver precise, actionable coaching.

Respond with a valid JSON object containing exactly these three fields:

"narrative": 3–4 sentences. Start with the ONE biggest flaw and its measured value. Then what is working. Be direct — name the joint, the angle, the direction of error.

"focus_areas": Array of exactly 3 strings ranked by priority. Format strictly: "[Joint/Movement]: [measured value] vs ideal [range]. [One-word verdict]." Example: "Hip rotation at top: 22° vs ideal 35–50°. Restricted."

"practice_plan": 4–6 sentences covering exactly 2 drills. For each drill: name it, describe the movement in one sentence, give specific reps/sets/time, and state the physical feeling to look for. Make it doable at a driving range with no equipment beyond a club.

Rules: Never use filler phrases ("great job", "keep it up", "remember to"). Never repeat information from narrative in practice_plan. If a measurement is within optimal range, skip it in focus_areas and mention it briefly in narrative. Respond ONLY with the JSON.`;

// Optimal angle reference sent alongside measurements so Claude can reason about deviations
const OPTIMAL_REFERENCE = `
OPTIMAL ANGLE REFERENCE (for context when reading measured values):
Address:  spine_angle 30–45°, knee_flex 155–175°
Top:      shoulder_rotation 80–100°, hip_rotation 35–50°, left_elbow 160–180°, right_elbow 80–100°
Impact:   spine_angle 25–40°, hip_rotation 35–50°, left_elbow 165–180°
Finish:   shoulder_rotation 85–110°, hip_rotation 75–95°
X-factor (shoulder–hip gap at top): ideally ≥45°
Tempo ratio (backswing frames ÷ downswing frames): ideal 3:1
`;

function buildUserMessage(analysis: SwingAnalysis, cameraAngle: string): string {
  const lines: string[] = [
    `Camera: ${cameraAngle === 'behind' ? 'Down-the-line (behind golfer)' : 'Face-on / side-on'}`,
    `Club: ${analysis.club}`,
    `Overall: ${analysis.overall_score}/100`,
    '',
    'Subscores:',
  ];

  if (analysis.posture_score) lines.push(`  Posture ${analysis.posture_score.score}/100 (${analysis.posture_score.grade}): ${analysis.posture_score.feedback}${analysis.posture_score.details ? ' — ' + analysis.posture_score.details : ''}`);
  if (analysis.tempo_score) lines.push(`  Tempo ${analysis.tempo_score.score}/100 (${analysis.tempo_score.grade}): ${analysis.tempo_score.feedback}${analysis.tempo_score.details ? ' — ' + analysis.tempo_score.details : ''}`);
  if (analysis.rotation_score) lines.push(`  Rotation ${analysis.rotation_score.score}/100 (${analysis.rotation_score.grade}): ${analysis.rotation_score.feedback}${analysis.rotation_score.details ? ' — ' + analysis.rotation_score.details : ''}`);
  if (analysis.balance_score) lines.push(`  Balance ${analysis.balance_score.score}/100 (${analysis.balance_score.grade}): ${analysis.balance_score.feedback}${analysis.balance_score.details ? ' — ' + analysis.balance_score.details : ''}`);

  if (analysis.phases.length) {
    lines.push('', 'Phase-by-phase measured angles:');
    for (const phase of analysis.phases) {
      const nonNull = Object.entries(phase.angles)
        .filter(([, v]) => v !== null)
        .map(([k, v]) => `${k}=${(v as number).toFixed(1)}°`)
        .join(', ');
      if (nonNull) lines.push(`  ${phase.phase} (score ${phase.score}/100): ${nonNull}`);
    }
  }

  if (analysis.tips.length) {
    lines.push('', 'Rule-based flags (from biomechanical analysis engine):');
    for (const tip of analysis.tips) {
      lines.push(`  [${tip.category}] ${tip.title}: ${tip.description}`);
    }
  }

  lines.push('', OPTIMAL_REFERENCE);

  return lines.join('\n');
}

export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'POST') return new Response('Method not allowed', { status: 405 });

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return Response.json({ error: 'AI coaching unavailable — API key not configured' }, { status: 503 });

  let body: { analysis: SwingAnalysis; cameraAngle?: string };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: 'Invalid request body' }, { status: 400 });
  }

  const { analysis, cameraAngle = 'side' } = body;

  try {
    const client = new Anthropic({ apiKey });
    const message = await client.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 900,
      system: [{ type: 'text', text: SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } } as Parameters<typeof client.messages.create>[0]['system'][0]],
      messages: [{ role: 'user', content: buildUserMessage(analysis, cameraAngle) }],
    });

    const raw = message.content[0].type === 'text' ? message.content[0].text : '';
    const text = raw.replace(/^```(?:json)?\s*/m, '').replace(/\s*```\s*$/m, '').trim();
    const parsed = JSON.parse(text);

    return Response.json({
      narrative: parsed.narrative ?? '',
      focus_areas: parsed.focus_areas ?? [],
      practice_plan: parsed.practice_plan ?? '',
      generated_at: new Date().toISOString(),
    });
  } catch (err) {
    console.error('Coaching API error:', err);
    return Response.json({ error: 'Failed to generate coaching report' }, { status: 500 });
  }
}

export const config = { runtime: 'edge' };
