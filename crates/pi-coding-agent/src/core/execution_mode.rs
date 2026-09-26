//! The model-facing execution surface, independent of presentation and autonomy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionMode {
    #[default]
    Ipython,
    Node,
    Direct,
}

impl ExecutionMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            "ipython" => Ok(Self::Ipython),
            "node" => Ok(Self::Node),
            "direct" => Ok(Self::Direct),
            _ => Err("Usage: /mode [ipython|node|direct|toggle]".into()),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ipython => "ipython",
            Self::Node => "node",
            Self::Direct => "direct",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Ipython => "IPython",
            Self::Node => "Node",
            Self::Direct => "Direct tools",
        }
    }

    pub fn toggled(self) -> Self {
        match self {
            Self::Ipython => Self::Node,
            Self::Node => Self::Direct,
            Self::Direct => Self::Ipython,
        }
    }

    pub fn from_tools(tools: &[String]) -> Option<Self> {
        if tools.iter().any(|name| name == "ipython") {
            Some(Self::Ipython)
        } else if tools.iter().any(|name| name == "node") {
            Some(Self::Node)
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
            .filter(|name| !matches!(name.as_str(), "ipython" | "node" | "bash" | "edit"))
            .cloned()
            .collect();
        tools.extend(match self {
            Self::Ipython => vec!["ipython".into()],
            Self::Node => vec!["node".into()],
            Self::Direct => vec!["bash".into(), "edit".into()],
        });
        tools
    }
}
