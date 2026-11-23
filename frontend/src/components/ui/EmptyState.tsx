import { type LucideIcon } from 'lucide-react';

interface EmptyStateProps {
  icon: LucideIcon;
  message: string;
}

export default function EmptyState({ icon: IconComponent, message }: EmptyStateProps) {
  return (
    <div className="empty-state">
      <IconComponent size={48} color="var(--text-muted)" />
      <p>{message}</p>
    </div>
  );
}
