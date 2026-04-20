import Anthropic from '@anthropic-ai/sdk';
import type { SwingAnalysis } from '../src/types';

const SYSTEM_PROMPT = `You are an elite PGA-level golf instructor. You receive biomechanical swing data and give short, direct feedback.

Your response must be a valid JSON object with exactly these three fields:

- "narrative": 3-4 SHORT sentences total. One sentence on what's working. Two sentences on the single biggest problem — name the specific angle/measurement. No fluff, no lengthy explanations.
- "focus_areas": array of exactly 3 strings, each under 20 words. Format: "[Issue]: [measured value] vs [optimal value]". Ranked by priority.
- "practice_plan": 2-3 sentences. One specific drill for priority #1. Include reps or time. Actionable enough to do at the range tomorrow.

Be blunt and specific. No padding. Respond ONLY with the JSON object.`;

function buildUserMessage(analysis: SwingAnalysis, cameraAngle: string): string {
  const lines: string[] = [
    `Camera angle: ${cameraAngle === 'behind' ? 'Down-the-line (behind golfer)' : 'Face-on / side-on'}`,
    `Club: ${analysis.club}`,
    `Overall: ${analysis.overall_score}/100`,
    '',
    'Subscores:',
  ];

  if (analysis.posture_score) lines.push(`  Posture ${analysis.posture_score.score} (${analysis.posture_score.grade}): ${analysis.posture_score.feedback}${analysis.posture_score.details ? ' — ' + analysis.posture_score.details : ''}`);
  if (analysis.tempo_score) lines.push(`  Tempo ${analysis.tempo_score.score} (${analysis.tempo_score.grade}): ${analysis.tempo_score.feedback}${analysis.tempo_score.details ? ' — ' + analysis.tempo_score.details : ''}`);
  if (analysis.rotation_score) lines.push(`  Rotation ${analysis.rotation_score.score} (${analysis.rotation_score.grade}): ${analysis.rotation_score.feedback}${analysis.rotation_score.details ? ' — ' + analysis.rotation_score.details : ''}`);
  if (analysis.balance_score) lines.push(`  Balance ${analysis.balance_score.score} (${analysis.balance_score.grade}): ${analysis.balance_score.feedback}${analysis.balance_score.details ? ' — ' + analysis.balance_score.details : ''}`);

  if (analysis.phases.length) {
    lines.push('', 'Phase angles:');
    for (const phase of analysis.phases) {
      const nonNull = Object.entries(phase.angles)
        .filter(([, v]) => v !== null)
        .map(([k, v]) => `${k}=${(v as number).toFixed(1)}°`)
        .join(', ');
      if (nonNull) lines.push(`  ${phase.phase} (score ${phase.score}): ${nonNull}`);
    }
  }

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
      max_tokens: 600,
      system: [{ type: 'text', text: SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } } as Parameters<typeof client.messages.create>[0]['system'][0]],
      messages: [{ role: 'user', content: buildUserMessage(analysis, cameraAngle) }],
    });

    const text = message.content[0].type === 'text' ? message.content[0].text : '';
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
