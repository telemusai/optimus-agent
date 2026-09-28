//! The model-facing execution surface, independent of presentation and autonomy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionMode {
    #[default]
    Ipython,
    Node,
    Clang,
    Direct,
}

impl ExecutionMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            "ipython" => Ok(Self::Ipython),
            "node" => Ok(Self::Node),
            "clang" | "clang-repl" => Ok(Self::Clang),
            "direct" => Ok(Self::Direct),
            _ => Err("Usage: /mode [ipython|node|clang|direct|cycle]".into()),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ipython => "ipython",
            Self::Node => "node",
            Self::Clang => "clang",
            Self::Direct => "direct",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Ipython => "IPython",
            Self::Node => "Node",
            Self::Clang => "Clang-Repl",
            Self::Direct => "Direct tools",
        }
    }

    pub fn toggled(self) -> Self {
        match self {
            Self::Ipython => Self::Node,
            Self::Node => Self::Clang,
            Self::Clang => Self::Direct,
            Self::Direct => Self::Ipython,
        }
    }

    /// Preserve the pre-Clang wire command for clients that know three modes.
    pub fn legacy_toggled(self) -> Self {
        match self { Self::Ipython => Self::Node, Self::Node | Self::Clang => Self::Direct, Self::Direct => Self::Ipython }
    }

    pub fn from_tools(tools: &[String]) -> Option<Self> {
        if tools.iter().any(|name| name == "ipython") {
            Some(Self::Ipython)
        } else if tools.iter().any(|name| name == "node") {
            Some(Self::Node)
        } else if tools.iter().any(|name| name == "clang") {
            Some(Self::Clang)
        } else if ["bash", "edit"]
            .iter()
            .all(|name| tools.iter().any(|tool| tool == name))
        {
            Some(Self::Direct)
        } else {
            None
        }
    }

    pub fn tools(self, current: &[String]) -> Vec<String> {
        let mut tools: Vec<_> = current
            .iter()
            .filter(|name| !matches!(name.as_str(), "ipython" | "node" | "clang" | "bash" | "edit" | "subagent" | "attach_image"))
            .cloned()
            .collect();
        tools.extend(match self {
            Self::Ipython => vec!["ipython".into()],
            Self::Node => vec!["node".into(), "subagent".into(), "attach_image".into()],
            Self::Clang => vec!["clang".into(), "subagent".into(), "attach_image".into()],
            Self::Direct => vec!["bash".into(), "edit".into(), "subagent".into(), "attach_image".into()],
        });
        tools
    }
}
