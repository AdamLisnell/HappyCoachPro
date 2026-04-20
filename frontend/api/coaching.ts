import Anthropic from '@anthropic-ai/sdk';
import type { SwingAnalysis } from '../src/types';

const SYSTEM_PROMPT = `You are an elite PGA-level golf instructor with 25 years of experience coaching players at every level. You receive structured biomechanical data from video swing analysis software and translate it into clear, actionable coaching.

Your response must be a valid JSON object with exactly these three fields:
- "narrative": 3-4 paragraphs. Start with what the golfer is doing well. Address the main weakness directly and specifically. Use professional but accessible language — explain any jargon. No generic praise.
- "focus_areas": array of exactly 3 strings. Each must name a specific issue with a specific measured angle or score (e.g. "Hip rotation at impact: 38° measured vs 45-50° optimal — hips not clearing through the ball"). Ranked by priority.
- "practice_plan": one paragraph. A single specific drill for the #1 issue. Include reps, sets, or time. Actionable enough to do at the range tomorrow.

Tone: Direct, knowledgeable, encouraging but honest. Treat the golfer as someone who wants real feedback, not flattery.

Respond ONLY with the JSON object, no markdown, no preamble.`;

function buildUserMessage(analysis: SwingAnalysis): string {
  const lines: string[] = [
    `Club: ${analysis.club}`,
    `Overall score: ${analysis.overall_score}/100`,
    '',
    '--- Subscores ---',
  ];

  if (analysis.posture_score) {
    lines.push(`Posture: ${analysis.posture_score.score}/100 (${analysis.posture_score.grade}) — ${analysis.posture_score.feedback}`);
    if (analysis.posture_score.details) lines.push(`  Detail: ${analysis.posture_score.details}`);
  }
  if (analysis.tempo_score) {
    lines.push(`Tempo: ${analysis.tempo_score.score}/100 (${analysis.tempo_score.grade}) — ${analysis.tempo_score.feedback}`);
    if (analysis.tempo_score.details) lines.push(`  Detail: ${analysis.tempo_score.details}`);
  }
  if (analysis.rotation_score) {
    lines.push(`Rotation: ${analysis.rotation_score.score}/100 (${analysis.rotation_score.grade}) — ${analysis.rotation_score.feedback}`);
    if (analysis.rotation_score.details) lines.push(`  Detail: ${analysis.rotation_score.details}`);
  }
  if (analysis.balance_score) {
    lines.push(`Balance: ${analysis.balance_score.score}/100 (${analysis.balance_score.grade}) — ${analysis.balance_score.feedback}`);
    if (analysis.balance_score.details) lines.push(`  Detail: ${analysis.balance_score.details}`);
  }

  if (analysis.phases.length) {
    lines.push('', '--- Phase Angles ---');
    for (const phase of analysis.phases) {
      lines.push(`${phase.phase.toUpperCase()} (score: ${phase.score}):`);
      const angles = phase.angles;
      const nonNull = Object.entries(angles)
        .filter(([, v]) => v !== null)
        .map(([k, v]) => `  ${k}: ${(v as number).toFixed(1)}°`);
      lines.push(...nonNull);
    }
  }

  return lines.join('\n');
}

export default async function handler(req: Request): Promise<Response> {
  if (req.method !== 'POST') {
    return new Response('Method not allowed', { status: 405 });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return Response.json(
      { error: 'AI coaching unavailable — API key not configured' },
      { status: 503 },
    );
  }

  let analysis: SwingAnalysis;
  try {
    analysis = await req.json();
  } catch {
    return Response.json({ error: 'Invalid request body' }, { status: 400 });
  }

  try {
    const client = new Anthropic({ apiKey });

    const message = await client.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 1024,
      system: [
        {
          type: 'text',
          text: SYSTEM_PROMPT,
          // @ts-expect-error — cache_control is valid but not yet in all SDK type defs
          cache_control: { type: 'ephemeral' },
        },
      ],
      messages: [{ role: 'user', content: buildUserMessage(analysis) }],
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
