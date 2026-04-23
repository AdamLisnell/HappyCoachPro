import Anthropic from '@anthropic-ai/sdk';
import type { SwingAnalysis, GolfClub } from '../src/types';

const SYSTEM_PROMPT = `You are an elite PGA-level golf instructor analyzing biomechanical swing data.

You MUST respond with a valid JSON object containing exactly these four fields:

"narrative": 3 sentences. Start with the ONE biggest flaw and its MEASURED value (e.g. "Hip rotation at top is restricted at 22° versus the 35–50° ideal"). Then note what is working well. Be direct — name the joint, the angle, the direction of error. No filler.

"ball_flight": 1 sentence. Explain what this swing likely produces as a ball flight outcome (slice, push, pull-hook, fat/thin contact, low launch, etc.) and WHY, based on the measurements. Example: "With an open face at impact and steep shoulder plane, expect a weak cut that loses distance right."

"focus_areas": Array of EXACTLY 3 strings ranked by priority. Format strictly: "[Joint/Movement]: [measured value] vs ideal [range]. [One-word verdict]." Example: "Hip rotation at top: 22° vs ideal 35–50°. Restricted."

"practice_plan": 4–6 sentences covering EXACTLY 2 drills. For each drill: name it, describe the movement in one sentence, give specific reps/sets/time, and state the physical feeling to look for. End with one session-end feel checkpoint ("By the last swing you should feel X"). Range-friendly, no equipment beyond a club.

Rules: Never use filler ("great job", "keep it up", "remember to"). Never repeat narrative content in practice_plan. Tailor advice to the club — driver needs upward angle of attack and wider arc, mid-iron needs ball-first contact and steeper plane, wedges need compact swing and hands-ahead impact. If a measurement is within optimal range, skip it in focus_areas. Respond ONLY with the JSON — no markdown, no prose.`;

const OPTIMAL_REFERENCE = `
OPTIMAL ANGLE REFERENCE:
Address:  spine_angle 30–45°, knee_flex 155–175°
Top:      shoulder_rotation 80–100°, hip_rotation 35–50°, left_elbow 160–180°, right_elbow 80–100°
Impact:   spine_angle 25–40°, hip_rotation 35–50°, left_elbow 165–180°
Finish:   shoulder_rotation 85–110°, hip_rotation 75–95°
X-factor at top (shoulder − hip): ideally ≥45°
Tempo ratio (backswing ÷ downswing frames): ideal 3:1`;

function clubContext(club: GolfClub): string {
  if (club === 'driver' || club === 'wood_3' || club === 'wood_5') {
    return 'Driver/Wood: expect wider arc, upward angle of attack, shoulder turn ≥90°, ball-first-on-upswing contact.';
  }
  if (club === 'putter') {
    return 'Putter: pendulum shoulder motion only, minimal hip/wrist action.';
  }
  if (club.startsWith('iron_') || club === 'hybrid') {
    return 'Mid/short iron: ball-first contact with small divot, steeper plane, 80–90° shoulder turn, hands ahead at impact.';
  }
  return 'Wedge: compact controlled swing, steep plane, hands well ahead at impact, limited lower-body rotation.';
}

function buildUserMessage(analysis: SwingAnalysis, cameraAngle: string): string {
  const lines: string[] = [
    `Camera: ${cameraAngle === 'behind' ? 'Down-the-line (behind golfer)' : 'Face-on / side-on'}`,
    `Club: ${analysis.club} — ${clubContext(analysis.club)}`,
    `Overall: ${analysis.overall_score}/100`,
  ];

  if (analysis.tempo_ratio !== undefined) lines.push(`Tempo ratio (BS:DS): ${analysis.tempo_ratio.toFixed(2)}:1`);
  if (analysis.x_factor_top !== undefined) lines.push(`X-factor at top: ${analysis.x_factor_top.toFixed(0)}°`);

  lines.push('', 'Subscores:');
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
    lines.push('', 'Rule-based flags:');
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
      max_tokens: 1100,
      system: [{ type: 'text', text: SYSTEM_PROMPT, cache_control: { type: 'ephemeral' } } as Parameters<typeof client.messages.create>[0]['system'][0]],
      messages: [{ role: 'user', content: buildUserMessage(analysis, cameraAngle) }],
    });

    const raw = message.content[0].type === 'text' ? message.content[0].text : '';
    const text = raw.replace(/^```(?:json)?\s*/m, '').replace(/\s*```\s*$/m, '').trim();
    const parsed = JSON.parse(text);

    return Response.json({
      narrative: parsed.narrative ?? '',
      ball_flight: parsed.ball_flight ?? '',
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
