import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface StreamingMessageProps {
  content: string;
  isStreaming?: boolean;
}

export const StreamingMessage = ({
  content,
  isStreaming = false,
}: StreamingMessageProps) => {
  return (
    <div className="text-xs leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: (props) => (
            <h1
              className="text-base font-bold text-foreground mb-3 pb-1 border-b"
              {...props}
            />
          ),
          h2: (props) => (
            <h2
              className="text-sm font-bold text-foreground mb-2 mt-4 pb-1 border-b border-border/50"
              {...props}
            />
          ),
          h3: (props) => (
            <h3
              className="text-xs font-semibold text-foreground mb-2 mt-3"
              {...props}
            />
          ),
          h4: (props) => (
            <h4
              className="text-xs font-semibold text-foreground/90 mb-1 mt-2"
              {...props}
            />
          ),
          p: (props) => (
            <p className="text-foreground leading-relaxed mb-2" {...props} />
          ),
          ul: (props) => (
            <ul
              className="list-disc list-outside space-y-0.5 mb-2 text-foreground pl-4"
              {...props}
            />
          ),
          ol: (props) => (
            <ol
              className="list-decimal list-outside space-y-0.5 mb-2 text-foreground pl-4"
              {...props}
            />
          ),
          li: (props) => (
            <li className="text-foreground leading-relaxed text-xs" {...props} />
          ),
          blockquote: (props) => (
            <blockquote
              className="border-l-4 border-accent bg-accent/10 px-4 py-2 my-3 italic text-foreground rounded-r-lg"
              {...props}
            />
          ),
          code: ({ className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || "");

            if (match) {
              return (
                <div className="my-2 rounded-lg overflow-hidden shadow-sm border">
                  <div className="bg-muted px-3 py-1 text-xs font-medium text-muted-foreground border-b">
                    {match[1]}
                  </div>
                  <pre className="bg-muted p-3 overflow-x-auto">
                    <code className="text-xs font-mono text-foreground">
                      {String(children).replace(/\n$/, "")}
                    </code>
                  </pre>
                </div>
              );
            } else {
              return (
                <code
                  className="bg-muted text-foreground px-1 py-0.5 rounded text-xs font-mono border"
                  {...props}
                >
                  {children}
                </code>
              );
            }
          },
          a: (props) => (
            <a
              className="text-accent hover:text-accent/80 underline decoration-accent/30 hover:decoration-accent/60 transition-colors font-medium"
              target="_blank"
              rel="noopener noreferrer"
              {...props}
            />
          ),
          em: (props) => (
            <em className="italic text-foreground font-medium" {...props} />
          ),
          strong: (props) => (
            <strong className="font-semibold text-foreground" {...props} />
          ),
          table: (props) => (
            <div className="my-4 overflow-x-auto">
              <table
                className="min-w-full divide-y divide-border border rounded-lg"
                {...props}
              />
            </div>
          ),
          thead: (props) => <thead className="bg-muted" {...props} />,
          th: (props) => (
            <th
              className="px-4 py-2 text-left text-sm font-semibold text-foreground border-b"
              {...props}
            />
          ),
          td: (props) => (
            <td
              className="px-4 py-2 text-sm text-foreground border-b border-border/50"
              {...props}
            />
          ),
          hr: (props) => (
            <hr
              className="my-6 border-0 h-px bg-gradient-to-r from-transparent via-border to-transparent"
              {...props}
            />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
      {isStreaming && (
        <span className="inline-block w-0.5 h-4 bg-accent ml-1 animate-pulse" />
      )}
    </div>
  );
};
