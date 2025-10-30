import { ExternalLink } from "lucide-react";

export interface Source {
  url: string;
  content?: string;
  favicon?: string;
  similarity_score?: number;
  rerank_score?: number;
  final_score?: number;
  metadata?: any;
}

interface SourcesFaviconsProps {
  sources: Source[];
  onSourceClick: (sources: Source[]) => void;
}

export const SourcesFavicons = ({ sources, onSourceClick }: SourcesFaviconsProps) => {
  if (!sources || sources.length === 0) return null;

  const displayedSources = sources.slice(0, 2);
  const remainingCount = sources.length - 2;

  const getFavicon = (source: Source) => {
    if (source.favicon) return source.favicon;
    try {
      const domain = new URL(source.url).hostname;
      return `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
    } catch {
      return `https://www.google.com/s2/favicons?domain=example.com&sz=32`;
    }
  };

  const getDomain = (url: string) => {
    try {
      return new URL(url).hostname.replace('www.', '');
    } catch {
      return url;
    }
  };

  return (
    <div className="flex items-center gap-1 sm:gap-2 mt-3 pt-3 border-t border-border/40">
      <button
        onClick={() => onSourceClick(sources)}
        className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 py-1.5 rounded-lg border border-border/60 hover:border-accent/60 hover:bg-accent/5 transition-all duration-200 group w-full sm:w-auto"
      >
        <ExternalLink className="h-3 w-3 sm:h-3.5 sm:w-3.5 text-muted-foreground group-hover:text-accent transition-colors flex-shrink-0" />
        <div className="flex items-center gap-1 sm:gap-1.5 min-w-0 flex-1 sm:flex-initial">
          {displayedSources.map((source, idx) => (
            <div key={idx} className="flex items-center gap-0.5 sm:gap-1 min-w-0">
              <img
                src={getFavicon(source)}
                alt=""
                className="h-3 w-3 sm:h-4 sm:w-4 rounded-sm flex-shrink-0"
                onError={(e) => {
                  e.currentTarget.src = `https://www.google.com/s2/favicons?domain=example.com&sz=32`;
                }}
              />
              <span className="text-xs text-muted-foreground group-hover:text-accent transition-colors truncate max-w-[60px] sm:max-w-[120px]">
                {getDomain(source.url)}
              </span>
            </div>
          ))}
          {remainingCount > 0 && (
            <span className="text-xs text-muted-foreground group-hover:text-accent transition-colors ml-0.5 flex-shrink-0">
              +{remainingCount}
            </span>
          )}
        </div>
      </button>
    </div>
  );
};
