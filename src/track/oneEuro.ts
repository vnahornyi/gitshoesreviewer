export type OneEuroParams = {
  minCutoff: number;
  beta: number;
  derivativeCutoff: number;
};

export type OneEuroState = {
  value: number;
  derivative: number;
  time: number;
};

function smoothing(cutoff: number, elapsed: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / elapsed);
}

export function startOneEuro(value: number, time: number): OneEuroState {
  return { value, derivative: 0, time };
}

// Casiez et al. 2012: the cutoff rises with speed, so slow jitter is smoothed and fast motion is not delayed.
export function stepOneEuro(
  state: OneEuroState,
  value: number,
  time: number,
  params: OneEuroParams,
): OneEuroState {
  const elapsed = (time - state.time) / 1000;
  if (elapsed <= 0) {
    return state;
  }
  const rawDerivative = (value - state.value) / elapsed;
  const derivativeAlpha = smoothing(params.derivativeCutoff, elapsed);
  const derivative =
    state.derivative + derivativeAlpha * (rawDerivative - state.derivative);
  const cutoff = params.minCutoff + params.beta * Math.abs(derivative);
  const alpha = smoothing(cutoff, elapsed);
  return {
    value: state.value + alpha * (value - state.value),
    derivative,
    time,
  };
}
