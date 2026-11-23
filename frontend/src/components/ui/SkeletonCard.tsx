interface SkeletonCardProps {
  height?: string;
}

export default function SkeletonCard({ height = '120px' }: SkeletonCardProps) {
  return (
    <div className="card glass skeleton-card" style={{ minHeight: height }}>
      <div className="skeleton skeleton-line" style={{ width: '40%', height: '12px', marginBottom: '16px' }} />
      <div className="skeleton skeleton-line" style={{ width: '60%', height: '32px', marginBottom: '12px' }} />
      <div className="skeleton skeleton-line" style={{ width: '80%', height: '8px' }} />
    </div>
  );
}
