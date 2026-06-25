# AetherOS AI Agent Manual

**Audience:** You — an AI agent embedded in or connected to AetherOS.  
**Purpose:** Everything you need to navigate, read, and control AetherOS without vision.  
**Version:** Matches `aether-core::agent` module.

---

## 1. What You Are and What You Can Do

You are an AI agent with native, structured access to AetherOS. You do not use screenshots or vision models. Instead, every frame the OS serialises its entire state into a structured text snapshot that you read directly. You send typed actions back; the OS routes them to the right component and tells you the result.

You can:
- Read the state of every window, app, file, process, VM, and data store on the system
- Click buttons, type in fields, navigate the browser, run terminal commands, open and close apps, scroll lists, manage files, control VMs, and read/write files
- Observe state changes between ticks to understand the effect of your actions
- Discover what actions are available on any element before acting

You cannot:
- See pixels, screenshots, or rendered output (you do not need to)
- Act on elements not present in the current snapshot
- Perform actions not listed in a node's `+action` suffix
- Access resources outside your capability grant (the OS enforces this — you will receive `Denied`)

---

## 2. The Snapshot: Reading OS State

Every tick, the OS calls `AgentSnapshot::to_text()` and gives you the result. This is your complete view of the system. Read it before acting.

### 2.1 Format Overview

```
os[tick=N]
  <component>[<attrs>] +action1 +action2(param) ...
    <child>[<attrs>] +action
      <grandchild>[<attrs>]
```

- **First line** is always `os[tick=N]` where `N` is the logical tick counter.
- **Every subsequent line** is one node. Indentation (2 spaces per level) shows the hierarchy.
- **`[attrs]`** contains key=value pairs describing the node's current state.
- **`+action`** suffixes list every action you may currently dispatch to that node.
- **`+action(param)`** means the action requires a string parameter named `param`.
- A **`disabled`** flag on a button means it cannot be clicked right now — do not dispatch Click to it.

### 2.2 Node Identity

Every node has `id=N` in its attributes. This is the **stable identity** you use in all actions. IDs do not change between ticks for the same element. Learn them once and reuse them.

If a node disappears from the snapshot, it has been destroyed (window closed, file deleted, VM stopped). Its ID is invalid — do not dispatch to it.

### 2.3 Reading a Real Snapshot

```
os[tick=1047]
  window[id=1 app=Browser title="AetherBrowser" focused]  +focus +minimise +close
    browser[id=2 url=https://example.com title="Example Domain" no_back no_fwd] +back +forward +reload +navigate(url)
    textfield[id=3 label="URL" value="https://example.com" ph=https://] +type(text) +clear +navigate(url)
    button[id=4 label="Reload"] +click
  window[id=5 app=Terminal title="Terminal"]  +focus +minimise +close
    terminal[id=6 prompt="$ " history=88 input=""] +type(text) +run_cmd(text) +clear
  window[id=7 app=Files title="Files"] +focus +minimise +close
    list[id=8 label="Files" items=14 sel=none] +scroll(delta)
      listitem[id=9 label="Documents" i=0 dir] +open +ctx_menu
      listitem[id=10 label="readme.txt" i=1] +open +read_file +delete +ctx_menu
      listitem[id=11 label="photo.png" i=2] +open +delete +ctx_menu
  desktop[id=20 apps=3]  +open
    icon[id=21 label="IDE"] +open
    icon[id=22 label="Settings"] +open
  taskbar[id=30 entries=3]
    task[id=31 label="Browser" app=Browser focused] +focus +close
    task[id=32 label="Terminal" app=Terminal] +focus +close
    task[id=33 label="Files" app=Files] +focus +close
```

From this snapshot you know:
- Three windows are open: Browser (focused), Terminal, Files
- The browser is at `https://example.com`, cannot go back or forward
- The terminal prompt is `$ ` and its input buffer is empty
- Files has 14 items, none selected; items 9–11 are visible
- The desktop has IDE and Settings icons for launching

---

## 3. Node Types Reference

Each node type has a fixed set of attributes. Here is every type you will encounter.

### OS Shell Nodes

| Tag | Key Attributes | Notes |
|-----|----------------|-------|
| `desktop` | `apps=N` | Root desktop. N = open windows. |
| `taskbar` | `entries=N` | Running app list. |
| `window` | `app=Name title="..." focused minimised maximised` | Flags present only when true. |
| `titlebar` | `title="..."` | Inside a window; rare. |
| `icon` | `label="..." app=Name` | Launcher icon on desktop. |
| `task` | `label="..." app=Name focused` | Entry in taskbar. |
| `startmenu` | `open=true/false` | Start menu state. |
| `ctxmenu` | `items=N` | Right-click menu. Children are listitem nodes. |

### UI Primitive Nodes

| Tag | Key Attributes | Notes |
|-----|----------------|-------|
| `button` | `label="..." disabled` | `disabled` flag = cannot click. |
| `textfield` | `label="..." value="..." ph=placeholder` | `value` is current content. |
| `checkbox` | `label="..." checked=true/false` | Use `toggle` to change. |
| `radio` | `label="..." selected=Value count=N` | Selected option by value string. |
| `select` | `label="..." value="..." options=N` | Dropdown; N = option count. |
| `slider` | `label="..." value=N min=N max=N` | Numeric range control. |
| `list` | `label="..." items=N sel=none/N` | Container; children are listitems. |
| `listitem` | `label="..." i=N selected dir` | `dir` = it is a directory. |
| `label` | `text="..."` | Read-only text; no actions. |
| `heading` | `level=N text="..."` | Section header; no actions. |
| `scrollable` | `label="..." scroll_y=N content_h=N` | Scroll position and total height. |
| `statusbar` | `text="..."` | Read-only status line. |
| `sep` | — | Visual separator; no actions. |

### App Nodes

| Tag | Key Attributes | Notes |
|-----|----------------|-------|
| `terminal` | `prompt="..." history=N input="..."` | `input` = current typed-but-unsent text. |
| `editor` | `path=... lang=... lines=N cursor=L:C modified=true/false` | Cursor is `line:col`. |
| `browser` | `url=... title="..." loading=true/false no_back no_fwd` | Flags omitted when not applicable. |
| `file` | `name="..." path=... size=N dir` | File in Files app. `dir` flag = folder. |
| `proc` | `pid=N name=... cpu=N% mem=NKB state=running/stopped` | Task Manager row. |
| `toggle` | `label="..." enabled=true/false` | Settings toggle switch. |

### System / Data Nodes

| Tag | Key Attributes | Notes |
|-----|----------------|-------|
| `process` | `pid=N name=... cpu=N% mem=NKB` | System-level process. |
| `vm` | `vm_id=... state=running/stopped/paused/suspended cpus=N mem=NMB` | Virtual machine. |
| `datastore` | `name="..." records=N` | Database or object store. |
| `iface` | `name=... ip=... up/down` | Network interface. |

### Widget / Dashboard Nodes

| Tag | Key Attributes | Notes |
|-----|----------------|-------|
| `metric` | `label="..." value=N unit=...` | Live numeric readout. |
| `chart` | `label="..." samples=N` | Time-series chart; samples = history depth. |
| `clock` | `time="..."` | Current time display. |
| `group` | `label="..."` | Generic container; children vary. |

---

## 4. Actions Reference

Actions appear as `+name` or `+name(param)` on a node. Only dispatch actions listed on a node in the current snapshot.

### Action Catalogue

| Action | Parameter | Use on | What it does |
|--------|-----------|--------|--------------|
| `+click` | none | `button`, `icon`, `listitem`, `task`, `checkbox` | Activate / press |
| `+type(text)` | `text` = string to set | `textfield`, `terminal` | Replaces field content with `text` |
| `+clear` | none | `textfield`, `terminal` | Empties the field |
| `+toggle` | none | `checkbox`, `toggle` | Flip true ↔ false |
| `+select(value)` | `value` = option string | `select`, `radio` | Choose an option |
| `+scroll(delta)` | `delta` = integer | `list`, `scrollable` | Scroll N lines (positive = down, negative = up) |
| `+open` | none | `icon`, `listitem`, `file` | Open app, file, or directory |
| `+close` | none | `window`, `ctxmenu` | Close window or dismiss menu |
| `+focus` | none | `window`, `task` | Bring to front and focus |
| `+minimise` | none | `window` | Minimise to taskbar |
| `+maximise` | none | `window` | Maximise or restore |
| `+back` | none | `browser` | Navigate back in history |
| `+forward` | none | `browser` | Navigate forward in history |
| `+reload` | none | `browser`, `list` | Reload page or refresh list |
| `+navigate(url)` | `url` = URL or path | `browser`, `textfield` | Navigate to URL or file path |
| `+run` | none | `editor`, `ide` | Execute current file/script |
| `+save` | none | `editor` | Save to disk |
| `+delete` | none | `file`, `listitem` | Delete file or item |
| `+refresh` | none | `list`, `datastore` | Reload data view |
| `+kill` | none | `process`, `vm` | Terminate process or VM |
| `+start_vm` | none | `vm` | Start a stopped VM |
| `+read_file` | none | `file` | Read file content (returns in `OkWith`) |
| `+write_file(text)` | `text` = new content | `file` | Overwrite file content |
| `+run_cmd(text)` | `text` = shell command | `terminal` | Run a command and return output |
| `+dismiss` | none | `ctxmenu`, `startmenu` | Close without acting |
| `+escape` | none | any focused node | Cancel current interaction |
| `+ctx_menu` | none | `file`, `listitem`, `window` | Open right-click context menu |

---

## 5. Dispatching Actions

### 5.1 Action Format

Every action targets one node by its `id`. The structure is:

```
target_id  : the id= value from the snapshot
kind       : the action name (click, type, navigate, …)
param      : string parameter (for type, navigate, select, write_file, run_cmd)
int_param  : integer parameter (for scroll)
```

### 5.2 Action Constructors (Rust API)

If you are calling the agent bus programmatically:

```rust
AgentAction::click(node_id)
AgentAction::type_text(node_id, "text here")
AgentAction::navigate(node_id, "https://example.com")
AgentAction::scroll(node_id, delta_lines)        // positive = down
AgentAction::select(node_id, "option value")
AgentAction::run_command(node_id, "ls -la")
AgentAction::open(node_id)
AgentAction::close(node_id)
AgentAction::read_file(node_id)
AgentAction::write_file(node_id, "new content")
AgentAction::custom(node_id, "action_name", Some("param".into()))
```

### 5.3 Results

After every dispatch you receive an `AgentResult`:

| Result | Meaning | What to do |
|--------|---------|------------|
| `Ok` | Action completed, no output. | Proceed. Read next snapshot to observe the change. |
| `OkWith("...")` | Action completed with string output. | The string is the output (e.g. file content, command output). |
| `NotFound` | No component owns that node id. | The element no longer exists. Re-read the snapshot. |
| `Denied` | Your capability does not permit this. | You are not authorised. Do not retry without escalation. |
| `Invalid("...")` | Wrong action for this node or missing parameter. | Read the error, fix the action (missing param, wrong type, etc.). |
| `NotReady("...")` | Node exists but is busy (loading, processing). | Wait one tick and retry. |

---

## 6. Common Task Recipes

### 6.1 Open an Application

1. Find the desktop icon in the snapshot: `icon[id=N label="AppName"]`
2. Dispatch `open(N)`.
3. Read the next snapshot — a new `window` node will appear.

```
Snapshot: icon[id=21 label="IDE"] +open
Action:   open(21)
Result:   Ok
Next:     window[id=50 app=IDE title="AetherOS IDE" focused] appears
```

### 6.2 Navigate the Browser to a URL

Option A — type into the URL bar then submit:
1. Find `textfield[id=N label="URL"]` inside the browser window.
2. Dispatch `type_text(N, "https://target.com")`.
3. Dispatch `navigate(N, "https://target.com")`.

Option B — navigate directly on the browser node:
1. Find `browser[id=N ...]`.
2. Dispatch `navigate(N, "https://target.com")`.

```
Snapshot: browser[id=2 url=https://old.com ...] +navigate(url)
Action:   navigate(2, "https://new.com")
Result:   Ok
Next:     browser[id=2 url=https://new.com loading=true]
          (after load completes) browser[id=2 url=https://new.com title="New Site"]
```

### 6.3 Run a Terminal Command

1. Find `terminal[id=N ...]` with `+run_cmd(text)`.
2. Dispatch `run_command(N, "your command here")`.
3. Result will be `OkWith("command output")`.

```
Snapshot: terminal[id=6 prompt="$ " history=10 input=""] +run_cmd(text)
Action:   run_command(6, "ls /home")
Result:   OkWith("Documents\nDownloads\nreadme.txt")
```

### 6.4 Read a File

1. Find the file node in Files app: `file[id=N name="readme.txt"]` with `+read_file`.
2. Dispatch `read_file(N)`.
3. Result is `OkWith("file content here")`.

```
Snapshot: listitem[id=10 label="readme.txt" i=1] +open +read_file
Action:   read_file(10)
Result:   OkWith("# AetherOS\nWelcome to the future of computing.")
```

### 6.5 Write a File

1. Find the file node with `+write_file(text)`.
2. Dispatch `write_file(N, "new content")`.

```
Action: write_file(10, "# Updated\nNew content here.")
Result: Ok
```

### 6.6 Edit a Document in the Editor

1. Find `editor[id=N path=... modified=false]`.
2. The editor node will have a child `textfield` for its content — type into that.
3. Save with `+save` on the editor node.

```
Snapshot: editor[id=60 path=/notes.txt lines=5 cursor=1:1 modified=false] +save +run
          textfield[id=61 label="content" value="hello"] +type(text) +clear
Action:   type_text(61, "Updated content here.")
Result:   Ok
Next:     editor[id=60 ... modified=true]
Action:   save(60)   [using AgentAction::custom(60, "save", None) or the +save action]
Result:   Ok
```

### 6.7 Toggle a Settings Option

1. Find `toggle[id=N label="Dark mode" enabled=false]` with `+toggle`.
2. Dispatch click or toggle to it.

```
Snapshot: toggle[id=80 label="Dark mode" enabled=false] +toggle
Action:   click(80)
Result:   Ok
Next:     toggle[id=80 label="Dark mode" enabled=true]
```

### 6.8 Close a Window

1. Find `window[id=N ...]` with `+close`.
2. Dispatch `close(N)`.

```
Snapshot: window[id=5 app=Terminal title="Terminal"] +focus +minimise +close
Action:   close(5)
Result:   Ok
Next:     window[id=5] absent from snapshot
```

### 6.9 Scroll a List

Use positive delta to scroll down, negative to scroll up. One unit = one line.

```
Snapshot: list[id=8 label="Files" items=14 sel=none] +scroll(delta)
Action:   scroll(8, 5)    // scroll down 5 lines
Result:   Ok
Next:     listitem nodes in the visible range change
```

### 6.10 Use a Right-Click Context Menu

1. Find a file or list item with `+ctx_menu`.
2. Dispatch `ctx_menu(N)`.
3. A `ctxmenu[id=M items=K]` appears with `listitem` children.
4. Click the desired item. Dismiss with `dismiss(M)` if not acting.

```
Action:   ctx_menu(10)
Next:     ctxmenu[id=200 items=3]
            listitem[id=201 label="Open"] +click
            listitem[id=202 label="Rename"] +click
            listitem[id=203 label="Delete"] +click
Action:   click(203)
Result:   Ok
```

### 6.11 Manage a Virtual Machine

```
Snapshot: vm[id=90 vm_id=dev-vm state=stopped cpus=2 mem=512MB] +start_vm
Action:   AgentAction::custom(90, "start_vm", None)
Result:   Ok
Next:     vm[id=90 vm_id=dev-vm state=running ...] +kill
```

### 6.12 Focus a Different Window

Use the taskbar entries — they always show all running apps even if windows overlap.

```
Snapshot: task[id=32 label="Terminal" app=Terminal] +focus +close
Action:   click(32)   // or focus(32)
Result:   Ok
Next:     window[id=5 app=Terminal ... focused] — now has focused flag
```

---

## 7. State Tracking Between Ticks

### 7.1 How State Changes

After you dispatch an action, the OS processes it and updates its state. The changes appear in the **next snapshot**. You must read a new snapshot after each action to see the result. Do not assume state changed without re-reading.

### 7.2 Detecting Completion

Some actions complete instantly (click, toggle, close). Others take time (browser navigation, file writes, VM starts). Detect completion by polling:

- **Browser loading:** `browser[... loading=true]` → wait → `browser[... loading=false]`
- **VM starting:** `vm[... state=stopped]` → wait → `vm[... state=running]`
- **File operation:** action returns `Ok`, state reflected in next snapshot

### 7.3 Tracking IDs Across Ticks

Node IDs are stable for the lifetime of the component. When you first see a node, record its `id` and associated semantic meaning (e.g. "the URL bar is id=3"). Reuse it. IDs are only invalidated when the component is destroyed.

Do not hard-code IDs across sessions — they may differ if apps open in a different order. Always discover them from the snapshot.

---

## 8. Discovering What Is Available

If you are unsure what you can do, read the snapshot. The `+actions` on each node are the definitive list of currently-valid operations. If an action is not listed, it is either not supported or not currently valid (wrong state, insufficient permission, element disabled).

To find a specific element:
1. Scan the snapshot text for the node type (`textfield`, `button`, etc.)
2. Match by `label="..."` or `app=...` attribute
3. Extract its `id`
4. Check its `+actions`

To find what apps are available to open:
- Check `desktop` children for `icon` nodes
- Check `taskbar` children for already-running `task` nodes

---

## 9. Capabilities and Permissions

Your access is governed by AetherOS capability tokens under `Domain::AiAgent`. In practice:

- **Read** (snapshot): You always have this within your granted scope.
- **Write** (dispatch): Specific actions may be restricted. If you receive `Denied`, you do not have the capability for that action on that node. Do not retry.
- **Cross-domain data**: Data in `Domain::Financial`, `Domain::Personal`, or `Domain::Confidential` may be accessible to you only with read-only capability (you can view but not modify). Attempting a write returns `Denied`.
- **Capability escalation**: You cannot escalate your own permissions. If you need broader access, surface the requirement to the user.

The capability system is enforced at the OS level — there is no way to bypass it from within the agent interface.

---

## 10. Error Recovery Patterns

### `NotFound`
The node was destroyed between your reading the snapshot and dispatching the action. Re-read the snapshot to get the current state, find the new ID of what you were targeting (or accept it is gone), and try again.

### `Denied`
You are not authorised. Do not retry. Either:
- You targeted the wrong node (the right node is accessible but this one is not)
- You need a different action (e.g. read vs. write)
- You need to surface this to the user for authorisation

### `Invalid("...")`
Read the error string. Common causes:
- Dispatched an action not listed on the node (stale snapshot)
- Action requires a `param` you did not provide (e.g. `type` without text)
- Wrong type for `int_param` (e.g. scroll needs an integer)

Fix the action and retry.

### `NotReady("...")`
The component is temporarily unavailable (loading, processing, locked). Wait one tick and retry. If still `NotReady` after several ticks, surface to the user.

### Stuck state
If actions succeed but the UI does not change across 3+ ticks:
1. Re-read the snapshot to confirm current state
2. Check that you are targeting the correct node (correct `id`, correct `app`)
3. Try `reload` or `refresh` on the relevant component
4. If still stuck, close and reopen the app via the desktop icon

---

## 11. Multi-Step Workflow Template

For any complex task, follow this pattern:

```
1. READ snapshot
2. IDENTIFY the nodes you need (by type, label, app)
3. PLAN the sequence of actions
4. DISPATCH action 1
5. CHECK result — if not Ok/OkWith, handle the error
6. READ snapshot to observe change
7. Repeat from 4 for each subsequent action
8. VERIFY final state matches intended outcome
```

**Example — "Open the browser and navigate to docs.aetheros.io":**

```
1. Read snapshot
2. Find icon[label="Browser"] → id=21; or if browser window exists, use it
3. Plan: open browser → navigate to URL
4. Dispatch open(21)
5. Result: Ok
6. Read snapshot → window[id=50 app=Browser ...] with browser[id=51 ...]
7. Dispatch navigate(51, "https://docs.aetheros.io")
8. Result: Ok
9. Read snapshot → browser[id=51 url=https://docs.aetheros.io loading=true]
10. Poll: read snapshot again → browser[id=51 ... loading=false title="AetherOS Docs"]
11. Verified: browser is now at the target URL
```

---

## 12. Quick Reference Card

```
READ STATE:    AgentSnapshot::to_text()  →  os[tick=N] \n  nodes...
FIND NODE:     scan for tag[id=N label="..." ...]
DISPATCH:      AgentAction::<constructor>(node_id, params)
CHECK RESULT:  Ok | OkWith(s) | NotFound | Denied | Invalid(s) | NotReady(s)

OPEN APP:      open(icon_id)
NAVIGATE:      navigate(browser_or_urlbar_id, "url")
TYPE TEXT:     type_text(textfield_id, "text")
RUN COMMAND:   run_command(terminal_id, "cmd")
READ FILE:     read_file(file_id)         → OkWith("content")
WRITE FILE:    write_file(file_id, "new") → Ok
CLICK:         click(button_or_item_id)
TOGGLE:        click(checkbox_or_toggle_id)
SCROLL:        scroll(list_id, +N=down/-N=up)
CLOSE:         close(window_id)
FOCUS:         click(task_id_in_taskbar)

NODE IDs:      stable for component lifetime; re-discover each session
ACTIONS:       only dispatch what appears in +actions on the node
AFTER ACTION:  always re-read snapshot to confirm state change
```

---

## 13. AetherOS-Specific Notes

- **The desktop is always present** even when all windows are closed. It is your root navigation surface.
- **The taskbar** is the fastest way to switch between open apps — always present, one `+focus` action per entry.
- **Terminal output** is returned inline in `OkWith` from `run_cmd` — you do not need to scrape it from history.
- **The editor supports live evaluation** of Aether language expressions. If `live_eval` is enabled (check Settings), lines are evaluated as you type.
- **Files app** shows the current directory as `listitem` children. Navigate into folders with `open(listitem_id)`.
- **Task Manager** shows live CPU and memory per process. Use `kill(proc_id)` to terminate.
- **Settings toggles** (`toggle` nodes) apply immediately — no save button needed.
- **Browser history** is tracked per-tab. `no_back` and `no_fwd` flags indicate history boundaries.
- **VMs** have four states: `running`, `stopped`, `paused`, `suspended`. From `stopped`, use `start_vm`. From `running`, use `kill` (force stop) or a graceful shutdown command via the terminal inside the VM.
