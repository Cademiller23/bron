type RobinsonPoint = {
  x: number;
  y: number;
};

export const ROBINSON_VIEWBOX = {
  left: -180,
  top: -91.296,
  width: 360,
  height: 182.592,
} as const;

export const ROBINSON_VIEWBOX_STRING = `${ROBINSON_VIEWBOX.left} ${ROBINSON_VIEWBOX.top} ${ROBINSON_VIEWBOX.width} ${ROBINSON_VIEWBOX.height}`;
export const ROBINSON_ASPECT_RATIO = ROBINSON_VIEWBOX.width / ROBINSON_VIEWBOX.height;

export function viewBoxToPercent(viewBoxX: number, viewBoxY: number): RobinsonPoint {
  return {
    x: ((viewBoxX - ROBINSON_VIEWBOX.left) / ROBINSON_VIEWBOX.width) * 100,
    y: ((viewBoxY - ROBINSON_VIEWBOX.top) / ROBINSON_VIEWBOX.height) * 100,
  };
}

/** Calibrated against public/world-map-robinson.svg (Natural Earth Robinson). */
export function mapViewBoxToPercent(viewBoxX: number, viewBoxY: number): RobinsonPoint {
  return viewBoxToPercent(viewBoxX, viewBoxY);
}
