type FootPoints = {
  heel: [number, number];
  toe: [number, number];
  score: number;
  ankle?: [number, number, number];
};

const HIDDEN: FootPoints = { heel: [0, 0], toe: [0, 0], score: 0 };
const NO_ANKLE = [0, 0, 0];

export function framePoints({
  left = HIDDEN,
  right = HIDDEN,
}: {
  left?: FootPoints;
  right?: FootPoints;
}): number[] {
  const foot = ({ heel, toe, score }: FootPoints) => ({
    toe: [...toe, score],
    smallToe: [toe[0], toe[1] + 0.01, score],
    heel: [...heel, score],
  });
  const l = foot(left);
  const r = foot(right);
  return [
    ...(left.ankle ?? NO_ANKLE),
    ...(right.ankle ?? NO_ANKLE),
    ...l.toe,
    ...l.smallToe,
    ...l.heel,
    ...r.toe,
    ...r.smallToe,
    ...r.heel,
  ];
}

// FootNet's eight points per foot, as the detector returns them: only the big toe, little toe and heel are set, which
// is what the shoe pose reads.
export function refinedPoints(
  feet: Array<{ toe: [number, number]; heel: [number, number] } | null>,
): number[] {
  return feet.flatMap(foot => {
    const empty = [0, 0, 0];
    if (!foot) {
      return Array.from({ length: 8 }, () => empty).flat();
    }
    const score = 0.9;
    return [
      [...foot.toe, score],
      empty,
      empty,
      empty,
      [foot.toe[0], foot.toe[1] + 0.01, score],
      [...foot.heel, score],
      empty,
      empty,
    ].flat();
  });
}
