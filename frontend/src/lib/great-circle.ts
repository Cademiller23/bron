export type GeoPoint = {
  latitude: number;
  longitude: number;
};

export function greatCirclePoints(
  start: GeoPoint,
  end: GeoPoint,
  segments = 64,
): [number, number][] {
  const toRad = (degrees: number) => (degrees * Math.PI) / 180;
  const toDeg = (radians: number) => (radians * 180) / Math.PI;

  const lat1 = toRad(start.latitude);
  const lon1 = toRad(start.longitude);
  const lat2 = toRad(end.latitude);
  const lon2 = toRad(end.longitude);

  const distance =
    2 *
    Math.asin(
      Math.sqrt(
        Math.sin((lat2 - lat1) / 2) ** 2 +
          Math.cos(lat1) * Math.cos(lat2) * Math.sin((lon2 - lon1) / 2) ** 2,
      ),
    );

  if (distance === 0) {
    return [[start.longitude, start.latitude]];
  }

  const points: [number, number][] = [];

  for (let index = 0; index <= segments; index += 1) {
    const fraction = index / segments;
    const a = Math.sin((1 - fraction) * distance) / Math.sin(distance);
    const b = Math.sin(fraction * distance) / Math.sin(distance);
    const x = a * Math.cos(lat1) * Math.cos(lon1) + b * Math.cos(lat2) * Math.cos(lon2);
    const y = a * Math.cos(lat1) * Math.sin(lon1) + b * Math.cos(lat2) * Math.sin(lon2);
    const z = a * Math.sin(lat1) + b * Math.sin(lat2);
    const lat = Math.atan2(z, Math.hypot(x, y));
    const lon = Math.atan2(y, x);
    points.push([toDeg(lon), toDeg(lat)]);
  }

  return points;
}

export function pointAlongGreatCircle(start: GeoPoint, end: GeoPoint, progress: number) {
  const clamped = Math.min(Math.max(progress, 0), 1);
  const points = greatCirclePoints(start, end, 64);
  const index = clamped * (points.length - 1);
  const lower = Math.floor(index);
  const upper = Math.min(lower + 1, points.length - 1);
  const fraction = index - lower;
  const [lon1, lat1] = points[lower];
  const [lon2, lat2] = points[upper];

  return {
    longitude: lon1 + (lon2 - lon1) * fraction,
    latitude: lat1 + (lat2 - lat1) * fraction,
  };
}

export function bearingBetween(start: GeoPoint, end: GeoPoint) {
  const toRad = (degrees: number) => (degrees * Math.PI) / 180;
  const lat1 = toRad(start.latitude);
  const lat2 = toRad(end.latitude);
  const dLon = toRad(end.longitude - start.longitude);
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x =
    Math.cos(lat1) * Math.sin(lat2) -
    Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);

  return (Math.atan2(y, x) * 180) / Math.PI;
}
