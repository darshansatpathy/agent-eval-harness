---
id: sync-troubleshooting
title: Fixing sync problems
---
If notes are not syncing, first check the sync status icon in the sidebar. A yellow icon means sync is paused; a red icon means sync failed.

To fix a failed sync, sign out and sign back in on the affected device. If that does not help, go to Settings > Advanced and choose Rebuild local cache. Rebuilding the cache does not delete any notes stored on the server.

Sync requires access to sync.nimbusnotes.example on port 443. Corporate firewalls that block this host cause sync to fail with error code E-1042.
