import { CSSProperties } from 'react';

interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  borderRadius?: string | number;
  className?: string;
  style?: CSSProperties;
  count?: number;
  gap?: number;
}

/**
 * Reusable shimmer skeleton loading placeholder.
 * Renders a pulsing gradient block matching the dimensions provided.
 * Use in place of content cards, charts, or text while data loads.
 */
export function Skeleton({
  width = '100%',
  height = 20,
  borderRadius = 8,
  className = '',
  style,
  count = 1,
  gap = 8,
}: SkeletonProps) {
  if (count > 1) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap }} role="status" aria-label="Loading content">
        {Array.from({ length: count }).map((_, i) => (
          <div
            key={i}
            className={`skeleton-shimmer ${className}`}
            style={{ width, height, borderRadius, ...style }}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className={`skeleton-shimmer ${className}`}
      style={{ width, height, borderRadius, ...style }}
      role="status"
      aria-label="Loading content"
    />
  );
}

/**
 * Card-shaped skeleton for dashboard metric cards.
 */
export function SkeletonCard({ className = '' }: { className?: string }) {
  return (
    <div className={`card ${className}`} style={{ padding: 20 }}>
      <Skeleton width={100} height={14} style={{ marginBottom: 12 }} />
      <Skeleton width={60} height={28} style={{ marginBottom: 8 }} />
      <Skeleton width="80%" height={12} />
    </div>
  );
}

/**
 * Chart-shaped skeleton for visualization placeholders.
 */
export function SkeletonChart({ height = 240 }: { height?: number }) {
  return (
    <div className="card" style={{ padding: 20 }}>
      <Skeleton width={140} height={16} style={{ marginBottom: 16 }} />
      <Skeleton width="100%" height={height} borderRadius={12} />
    </div>
  );
}

export default Skeleton;
