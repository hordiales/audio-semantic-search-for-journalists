# MCP Server Troubleshooting

## JSON Parse Errors in Claude Desktop Logs

If you see errors like:
```
Unexpected token '🎵', "🎵 Audio S"... is not valid JSON
```

This means the startup script is outputting visual text that Claude Desktop is trying to parse as JSON.

### Solution Applied

The startup scripts now detect when they're being called by Claude Desktop (non-terminal mode) and suppress visual output:

- `start_mcp_uv.sh` - Uses `if [ -t 1 ]` to detect terminal mode
- `start_uv.py` - Uses `sys.stdout.isatty()` to detect terminal mode  
- `server.py` - Only prints status messages in terminal mode

### Testing

1. **Terminal mode** (shows visual output):
   ```bash
   ./start_mcp_uv.sh --dataset-dir ../dataset
   ```

2. **MCP mode** (silent output, used by Claude Desktop):
   ```bash
   ./start_mcp_uv.sh --dataset-dir ../dataset < /dev/null
   ```

### Restart Required

After updating the scripts:
1. Restart Claude Desktop completely
2. Check logs at: `/Users/[username]/Library/Logs/Claude/mcp-server-audio-semantic-search-master.log`
3. Look for successful connection without JSON parse errors

### Expected Behavior

- ✅ Server should start silently when called by Claude Desktop
- ✅ No JSON parse errors in logs
- ✅ Tools should appear in Claude Desktop interface
- ✅ First load may take 1-2 minutes (ML model loading)