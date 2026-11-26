import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Database, Trash2, RefreshCw, Eye, Download, X, Search } from 'lucide-react';
import apiClient from '../../services/api';
import { Skeleton } from '../ui/Skeleton';
import type { ScanHistoryItem } from '../../types';

const severityColor = (s: number) => s > 70 ? 'var(--danger)' : s > 40 ? 'var(--warning)' : 'var(--success)';

interface HistoryTabProps {
  addToast: (message: string, type: 'success' | 'error' | 'warning' | 'info', duration?: number) => void;
}

export default function HistoryTab({ addToast }: HistoryTabProps) {
  const [scans, setScans] = useState<ScanHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [selected, setSelected] = useState<ScanHistoryItem | null>(null);
  const [filter, setFilter] = useState<'all' | 'field' | 'leaf'>('all');
  const [search, setSearch] = useState('');
  const [dateFilter, setDateFilter] = useState<'all' | '7d' | '30d' | '90d'>('all');

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setLoadError(false);
    try {
      const { data } = await apiClient.get<ScanHistoryItem[]>('/history?limit=100');
      setScans(data);
    } catch {
      setLoadError(true);
      addToast('Could not load scan history.', 'error');
    } finally {
      setLoading(false);
    }
  }, [addToast]);

  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  const deleteScan = async (id: string) => {
    try {
      await apiClient.delete(`/history/${id}`);
      setScans((prev) => prev.filter((s) => s.id !== id));
      if (selected?.id === id) setSelected(null);
      addToast('Scan record deleted.', 'success', 3000);
    } catch {
      addToast('Failed to delete scan.', 'error');
    }
  };

  const handleSelect = async (scan: ScanHistoryItem) => {
    try {
      const { data } = await apiClient.get<ScanHistoryItem>(`/history/${scan.id}`);
      setSelected(data);
    } catch {
      addToast('Could not load full scan details.', 'error');
      setSelected(scan);
    }
  };

  const exportHistoryJSON = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(scans, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `agripulse_scan_history_${new Date().toISOString().slice(0,10)}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    addToast('History exported successfully as JSON.', 'success');
  };

  const filtered = useMemo(() => {
    const now = Date.now();
    const dayMs = 86400000;
    const cutoff = dateFilter === '7d' ? now - 7 * dayMs
      : dateFilter === '30d' ? now - 30 * dayMs
      : dateFilter === '90d' ? now - 90 * dayMs
      : 0;

    return scans.filter((s) => {
      if (filter !== 'all' && s.scan_type !== filter) return false;
      if (cutoff && new Date(s.timestamp).getTime() < cutoff) return false;
      if (search.trim()) {
        const q = search.toLowerCase();
        return (
          s.disease_type?.toLowerCase().includes(q) ||
          s.filename?.toLowerCase().includes(q) ||
          s.scan_type?.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [scans, filter, search, dateFilter]);

  return (
    <motion.div key="history" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="tab-viewport">
      <div className="history-layout">
        {/* Main table card */}
        <div className="card history-main-card">
          <div className="history-toolbar">
            <h3 className="history-toolbar__title">
              <Database size={20} color="var(--primary)" />
              Scan History
              <span className="history-toolbar__count">({filtered.length} records)</span>
            </h3>
            <div className="history-toolbar__actions">
              {/* Search */}
              <div className="history-search-wrapper">
                <Search size={14} />
                <input
                  type="text"
                  className="history-search-bar"
                  placeholder="Search disease, file…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  aria-label="Search scan history"
                />
                {search && (
                  <button onClick={() => setSearch('')} className="history-search-clear" aria-label="Clear search">
                    <X size={12} />
                  </button>
                )}
              </div>

              {/* Date filter */}
              <div className="history-filter-group">
                {(['all', '7d', '30d', '90d'] as const).map((d) => (
                  <button
                    key={d}
                    onClick={() => setDateFilter(d)}
                    className={`btn-icon ${dateFilter === d ? 'active' : ''}`}
                  >
                    {d === 'all' ? 'All time' : `Last ${d}`}
                  </button>
                ))}
              </div>

              {/* Type filter */}
              <div className="history-filter-group">
                {(['all', 'field', 'leaf'] as const).map((f) => (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className={`btn-icon ${filter === f ? 'active' : ''}`}
                    aria-pressed={filter === f}
                  >
                    {f.charAt(0).toUpperCase() + f.slice(1)}
                  </button>
                ))}
              </div>

              <button className="btn-icon" onClick={fetchHistory} title="Refresh" aria-label="Refresh History">
                <RefreshCw size={14} />
              </button>
              {scans.length > 0 && (
                <button className="btn-icon" onClick={exportHistoryJSON} title="Export JSON">
                  <Download size={13} /> Export
                </button>
              )}
            </div>
          </div>

          {loadError && scans.length > 0 && (
            <p role="alert" className="history-load-error">Could not refresh scan history. Previously loaded records are still shown.</p>
          )}
          {loading ? (
            <div className="history-skeleton" role="status" aria-label="Loading scan history" aria-busy="true">
              <Skeleton width="100%" height={36} borderRadius={8} />
              <Skeleton width="100%" height={36} borderRadius={8} />
              <Skeleton width="100%" height={36} borderRadius={8} />
              <Skeleton width="100%" height={36} borderRadius={8} />
            </div>
          ) : loadError && scans.length === 0 ? (
            <div className="empty-state history-empty" role="alert">
              <Database size={40} color="var(--text-muted)" style={{ opacity: 0.7, marginBottom: 8 }} />
              <h3>Scan history unavailable</h3>
              <p>Could not load saved records. Check your connection and try again.</p>
              <button className="btn-icon" onClick={fetchHistory}>Try again</button>
            </div>
          ) : filtered.length === 0 ? (
            <div className="empty-state history-empty">
              <Database size={40} color="var(--primary)" style={{ opacity: 0.7, marginBottom: 8 }} />
              <h3>No Scan Records Found</h3>
              <p>Run a field or leaf diagnosis to build your historical crop health log.</p>
            </div>
          ) : (
            <div className="history-table-wrapper">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Type</th>
                    <th>File</th>
                    <th>Diagnosis</th>
                    <th>Severity</th>
                    <th>Confidence</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  <AnimatePresence>
                    {filtered.map((scan) => (
                      <motion.tr
                        key={scan.id}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 10 }}
                        className={`history-row ${selected?.id === scan.id ? 'selected' : ''}`}
                        onClick={() => handleSelect(scan)}
                      >
                        <td className="history-cell-date">
                          {new Date(scan.timestamp).toLocaleDateString()}{' '}
                          {new Date(scan.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </td>
                        <td>
                          <span className={`scan-type-badge ${scan.scan_type}`}>{scan.scan_type}</span>
                        </td>
                        <td className="history-cell-filename">{scan.filename}</td>
                        <td className="history-cell-disease" style={{ color: scan.disease_type === 'Healthy' ? 'var(--success)' : 'var(--danger)' }}>
                          {scan.disease_type}
                        </td>
                        <td>
                          <span style={{ fontWeight: 700, color: severityColor(scan.severity) }}>
                            {scan.severity.toFixed(1)}%
                          </span>
                        </td>
                        <td className="history-cell-confidence">{scan.confidence.toFixed(1)}%</td>
                        <td>
                          <div className="history-cell-actions">
                            <button className="btn-icon" onClick={(e) => { e.stopPropagation(); handleSelect(scan); }} title="View details"><Eye size={12} /></button>
                            <button className="btn-icon history-delete-btn" onClick={(e) => { e.stopPropagation(); deleteScan(scan.id); }} title="Delete record"><Trash2 size={12} /></button>
                          </div>
                        </td>
                      </motion.tr>
                    ))}
                  </AnimatePresence>
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Detail panel */}
        <AnimatePresence>
          {selected && (
            <motion.div
              className="card history-detail-panel"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
            >
              <div className="history-detail-header">
                <h3 className="history-detail-title">Scan Details</h3>
                <button className="btn-icon" onClick={() => setSelected(null)}>
                  <X size={14} />
                </button>
              </div>
              <div className="history-detail-rows">
                {([
                  ['Type', selected.scan_type.toUpperCase()],
                  ['Filename', selected.filename],
                  ['Diagnosis', selected.disease_type],
                  ['Severity', `${selected.severity.toFixed(1)}%`],
                  ['Confidence', `${selected.confidence.toFixed(1)}%`],
                  ['Spread Risk', selected.spread_risk || 'N/A'],
                  ['Timestamp', new Date(selected.timestamp).toLocaleString()],
                ] as [string, string][]).map(([label, value]) => (
                  <div key={label} className="history-detail-row">
                    <span className="history-detail-row__label">{label}</span>
                    <span className="history-detail-row__value">{value}</span>
                  </div>
                ))}
                <div className="history-treatment-block">
                  <p className="history-treatment-label">Treatment Guideline</p>
                  <p className="history-treatment-text">{selected.treatment}</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
