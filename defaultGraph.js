export const defaultGraph = {
  last_node_id: 684,
  last_link_id: 1687,
  nodes: [
    {
      id: 19,
      type: "SeargePreviewImage",
      pos: [1140, 40],
      size: {
        0: 520,
        1: 540,
      },
      flags: {
        pinned: true,
      },
      order: 101,
      mode: 0,
      inputs: [
        {
          name: "images",
          type: "IMAGE",
          link: 1399,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "images",
          type: "IMAGE",
          links: null,
          shape: 3,
        },
      ],
      title: "Preview Image",
      properties: {
        "Node name for S&R": "SeargePreviewImage",
      },
      widgets_values: [true],
    },
    {
      id: 463,
      type: "SeargePreviewImage",
      pos: [1670, 40],
      size: {
        0: 520,
        1: 540,
      },
      flags: {
        pinned: true,
      },
      order: 109,
      mode: 0,
      inputs: [
        {
          name: "images",
          type: "IMAGE",
          link: 1400,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "images",
          type: "IMAGE",
          links: null,
          shape: 3,
        },
      ],
      title: "High Resolution Preview Image",
      properties: {
        "Node name for S&R": "SeargePreviewImage",
      },
      widgets_values: [true],
    },
    {
      id: 87,
      type: "SeargePreviewImage",
      pos: [-1450, -970],
      size: {
        0: 900,
        1: 1570,
      },
      flags: {
        pinned: true,
      },
      order: 93,
      mode: 0,
      inputs: [
        {
          name: "images",
          type: "IMAGE",
          link: 1404,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "images",
          type: "IMAGE",
          links: [1594],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Vertical Preview Image",
      properties: {
        "Node name for S&R": "SeargePreviewImage",
      },
      widgets_values: [false],
    },
    {
      id: 86,
      type: "SeargePreviewImage",
      pos: [50, -970],
      size: {
        0: 1600,
        1: 870,
      },
      flags: {
        pinned: true,
      },
      order: 97,
      mode: 0,
      inputs: [
        {
          name: "images",
          type: "IMAGE",
          link: 1594,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "images",
          type: "IMAGE",
          links: null,
          shape: 3,
        },
      ],
      title: "Horizontal Preview Image",
      properties: {
        "Node name for S&R": "SeargePreviewImage",
      },
      widgets_values: [false],
    },
    {
      id: 498,
      type: "Reroute",
      pos: [850, -1050],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 98,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1402,
          pos: [37.5, 0],
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1399],
          slot_index: 0,
          label: "img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 499,
      type: "Reroute",
      pos: [1250, -1050],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 107,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1401,
          pos: [37.5, 0],
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1400],
          slot_index: 0,
          label: "hr img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 268,
      type: "LoadImage",
      pos: [-450, 40],
      size: {
        0: 400,
        1: 470,
      },
      flags: {
        pinned: true,
      },
      order: 0,
      mode: 0,
      outputs: [
        {
          name: "IMAGE",
          type: "IMAGE",
          links: [1587],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "MASK",
          type: "MASK",
          links: null,
          shape: 3,
        },
      ],
      title: "Image-to-Image and Inpainting Source Image",
      properties: {
        "Node name for S&R": "LoadImage",
      },
      widgets_values: ["example.png", "image"],
      color: "#432",
      bgcolor: "#653",
    },
    {
      id: 269,
      type: "LoadImage",
      pos: [-450, 590],
      size: {
        0: 400,
        1: 470,
      },
      flags: {
        pinned: true,
      },
      order: 1,
      mode: 0,
      outputs: [
        {
          name: "IMAGE",
          type: "IMAGE",
          links: null,
          shape: 3,
        },
        {
          name: "MASK",
          type: "MASK",
          links: [1588],
          shape: 3,
        },
      ],
      title: "Inpainting Mask - right click -> MaskEditor",
      properties: {
        "Node name for S&R": "LoadImage",
      },
      widgets_values: ["example.png", "image"],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 3,
      type: "SeargeTextInputV2",
      pos: [10, 40],
      size: {
        0: 460,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 2,
      mode: 0,
      outputs: [
        {
          name: "prompt",
          type: "SRG_PROMPT_TEXT",
          links: [1577],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Main Prompt",
      properties: {
        "Node name for S&R": "MainPrompt",
      },
      widgets_values: [
        "a close up of a man with long hair and a beard, by Johannes Sveinsson Kjarval, dynamic cinematic lighting, portrait of a rugged warrior, color graded, nordic",
      ],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 502,
      type: "Reroute",
      pos: [-1600, -1100],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 88,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1483,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1404, 1597],
          slot_index: 0,
          label: "img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 503,
      type: "Reroute",
      pos: [-1600, -1150],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 103,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1484,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1407],
          label: "up img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 500,
      type: "Reroute",
      pos: [700, -1100],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 94,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1597,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1402],
          label: "img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 497,
      type: "Reroute",
      pos: [1100, -1150],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 105,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1407,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1401],
          label: "hr img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 5,
      type: "SeargeTextInputV2",
      pos: [10, 250],
      size: {
        0: 460,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 3,
      mode: 0,
      outputs: [
        {
          name: "prompt",
          type: "SRG_PROMPT_TEXT",
          links: [1578],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Secondary Prompt",
      properties: {
        "Node name for S&R": "SecondaryPrompt",
      },
      widgets_values: [
        "close up, man with long hair and a beard, by Johannes Sveinsson Kjarval, dynamic cinematic lighting, portrait of a rugged warrior, color graded, nordic",
      ],
      color: "#2a363b",
      bgcolor: "#3f5159",
    },
    {
      id: 6,
      type: "SeargeTextInputV2",
      pos: [10, 460],
      size: {
        0: 460,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 4,
      mode: 0,
      outputs: [
        {
          name: "prompt",
          type: "SRG_PROMPT_TEXT",
          links: [1579],
          shape: 3,
        },
      ],
      title: "Style - can use <prompt> as a placeholder",
      properties: {
        "Node name for S&R": "StylePrompt",
      },
      widgets_values: [
        "cinematic photo of <prompt>. highly detailed, professional, gritty, sharp focus, high budget hollywood movie",
      ],
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 7,
      type: "SeargeTextInputV2",
      pos: [10, 680],
      size: {
        0: 460,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 5,
      mode: 0,
      outputs: [
        {
          name: "prompt",
          type: "SRG_PROMPT_TEXT",
          links: [1580],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Negative Prompt",
      properties: {
        "Node name for S&R": "NegativePrompt",
      },
      widgets_values: ["dirty, out of focus"],
      color: "#322",
      bgcolor: "#533",
    },
    {
      id: 8,
      type: "SeargeTextInputV2",
      pos: [10, 890],
      size: {
        0: 460,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 6,
      mode: 0,
      outputs: [
        {
          name: "prompt",
          type: "SRG_PROMPT_TEXT",
          links: [1581, 1582],
          shape: 3,
        },
      ],
      title: "Negative Secondary Prompt and Style",
      properties: {
        "Node name for S&R": "NegativeSecondaryAndStylePrompt",
      },
      widgets_values: ["mud"],
      color: "#332922",
      bgcolor: "#593930",
    },
    {
      id: 524,
      type: "SeargeMagicBox",
      pos: [-2500, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 53,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1620,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1622],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      title: "Apply Loras",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["apply loras", "data stream", "data stream"],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 525,
      type: "SeargeMagicBox",
      pos: [-2100, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 54,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1622,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1623],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["none - skip", "data stream", "data stream"],
    },
    {
      id: 526,
      type: "SeargeMagicBox",
      pos: [-1700, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 55,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1623,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1624],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      title: "Prompt Styling",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["prompt styling", "data stream", "data stream"],
      color: "#322",
      bgcolor: "#533",
    },
    {
      id: 528,
      type: "SeargeMagicBox",
      pos: [-900, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 57,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1626,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1627],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      title: "CLIP Mixing",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["clip mixing", "data stream", "data stream"],
      color: "#322",
      bgcolor: "#533",
    },
    {
      id: 529,
      type: "SeargeMagicBox",
      pos: [-500, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 58,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1627,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1628],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      title: "Apply Controlnet and Revision",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["apply controlnet", "data stream", "data stream"],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 530,
      type: "SeargeMagicBox",
      pos: [-100, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 59,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1628,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1629],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: [],
          shape: 3,
          slot_index: 1,
        },
      ],
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["none - skip", "data stream", "data stream"],
    },
    {
      id: 531,
      type: "SeargeMagicBox",
      pos: [300, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 60,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1629,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1631],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      title: "Latent Inputs",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["latent inputs", "data stream", "data stream"],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 532,
      type: "SeargeMagicBox",
      pos: [700, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 61,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1631,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1632],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: [],
          shape: 3,
          slot_index: 1,
        },
      ],
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["none - skip", "data stream", "data stream"],
    },
    {
      id: 533,
      type: "SeargeMagicBox",
      pos: [1100, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 62,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1632,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1635],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      title: "Sampling",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["sampling", "data stream", "data stream"],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 536,
      type: "SeargeMagicBox",
      pos: [2300, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 65,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1641,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1647],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      title: "High Resolution",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["high resolution", "data stream", "data stream"],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 506,
      type: "SeargeMagicBox",
      pos: [-2900, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 52,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1619,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1620],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      title: "Load Checkpoints",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["load checkpoints", "data stream", "data stream"],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 537,
      type: "SeargeMagicBox",
      pos: [-3300, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 51,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1665,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1619],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      title: "Pre-Process Data",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["pre-process data", "data stream", "data stream"],
      color: "#2a363b",
      bgcolor: "#3f5159",
    },
    {
      id: 538,
      type: "SeargeMagicBox",
      pos: [2700, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 67,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1647,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1653],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: [],
          shape: 3,
          slot_index: 1,
        },
      ],
      title: "Hires Detailer",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["hires detailer", "data stream", "data stream"],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 539,
      type: "SeargeMagicBox",
      pos: [3100, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 69,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1653,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
          slot_index: 1,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1651],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: [1682],
          shape: 3,
          slot_index: 1,
        },
      ],
      title: "VAE Decode Hi-Res",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: [
        "vae decode hi-res",
        "data stream",
        "custom stage & data stream",
      ],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 540,
      type: "SeargeMagicBox",
      pos: [3500, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 71,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1651,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1650],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["none - skip", "data stream", "data stream"],
    },
    {
      id: 541,
      type: "SeargeMagicBox",
      pos: [3900, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 74,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1650,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1654],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: [1658],
          shape: 3,
          slot_index: 1,
        },
      ],
      title: "Upscaling",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: [
        "upscaling",
        "data stream",
        "custom stage & data stream",
      ],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 542,
      type: "SeargeMagicBox",
      pos: [4300, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 77,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1654,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1655],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["none - skip", "data stream", "data stream"],
    },
    {
      id: 543,
      type: "SeargeMagicBox",
      pos: [4700, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 81,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1655,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1656],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      title: "Image Saving",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["image saving", "data stream", "data stream"],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 544,
      type: "SeargeMagicBox",
      pos: [5100, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 85,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1656,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1458, 1657],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["none - skip", "data stream", "data stream"],
    },
    {
      id: 551,
      type: "Reroute",
      pos: [5400, 2930],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 89,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1458,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "SRG_DATA_STREAM",
          links: [],
          slot_index: 0,
          label: "debug",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 519,
      type: "SeargeDebugPrinter",
      pos: [5500, 2930],
      size: {
        0: 300,
        1: 120,
      },
      flags: {
        pinned: true,
      },
      order: 7,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: null,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: null,
          shape: 3,
        },
      ],
      properties: {
        "Node name for S&R": "SeargeDebugPrinter",
      },
      widgets_values: [false, "DONE"],
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 559,
      type: "Reroute",
      pos: [-5700, 3600],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 73,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1478,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1472],
          label: "img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 560,
      type: "Reroute",
      pos: [-5600, 3500],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 76,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1472,
          slot_index: 0,
          pos: [37.5, 0],
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1473],
          label: "img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 564,
      type: "Reroute",
      pos: [-5900, 3650],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 83,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1480,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1481],
          label: "hr img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 565,
      type: "Reroute",
      pos: [-5700, 3500],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 87,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1481,
          slot_index: 0,
          pos: [37.5, 0],
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1482],
          label: "hr img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 568,
      type: "Reroute",
      pos: [-5300, -1150],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 100,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1489,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1484],
          label: "hr img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 567,
      type: "Reroute",
      pos: [-5300, -1100],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 84,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1487,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1483],
          label: "img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 566,
      type: "Reroute",
      pos: [-5700, -1400],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 92,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1482,
          slot_index: 0,
          pos: [37.5, 0],
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1488],
          label: "hr img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 569,
      type: "Reroute",
      pos: [-5600, -1350],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 96,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1488,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1489],
          slot_index: 0,
          label: "hr img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 571,
      type: "Reroute",
      pos: [4600, 3550],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 82,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1659,
          pos: [37.5, 0],
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1613],
          label: "up img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 590,
      type: "SeargePipelineTerminator",
      pos: [5500, 3130],
      size: {
        0: 300,
        1: 70,
      },
      flags: {
        pinned: true,
      },
      order: 90,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1657,
          slot_index: 0,
        },
      ],
      properties: {
        "Node name for S&R": "SeargePipelineTerminator",
      },
      color: "#233",
      bgcolor: "#355",
    },
    {
      id: 610,
      type: "SeargeGenerationParameters",
      pos: [480, 40],
      size: {
        0: 320,
        1: 300,
      },
      flags: {
        pinned: true,
      },
      order: 27,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1566,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1567],
          shape: 3,
        },
      ],
      title: "Generation Parameters",
      properties: {
        "Node name for S&R": "SeargeGenerationParameters",
      },
      widgets_values: [
        56421668259,
        "increment",
        "1344x768 (16:9)",
        1024,
        1024,
        30,
        5,
        "1 - DPM++ 2M Karras",
        "dpmpp_2m",
        "karras",
        0.8,
      ],
      color: "#432",
      bgcolor: "#653",
    },
    {
      id: 612,
      type: "SeargeImageSaving",
      pos: [810, 40],
      size: {
        0: 320,
        1: 300,
      },
      flags: {
        pinned: true,
      },
      order: 30,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1567,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1568],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Image Saving",
      properties: {
        "Node name for S&R": "SeargeImageSaving",
      },
      widgets_values: [
        false,
        "output/Searge-SDXL-%date%",
        true,
        true,
        "generated-%seed%",
        true,
        true,
        "high-res-%seed%",
        true,
        true,
        "upscaled-%seed%",
      ],
      color: "#332922",
      bgcolor: "#593930",
    },
    {
      id: 611,
      type: "SeargeOperatingMode",
      pos: [480, 380],
      size: {
        0: 320,
        1: 106,
      },
      flags: {
        pinned: true,
      },
      order: 24,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1565,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1566],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Operating Mode",
      properties: {
        "Node name for S&R": "SeargeOperatingMode",
      },
      widgets_values: ["text-to-image", "default - all prompts", 1],
      color: "#322",
      bgcolor: "#533",
    },
    {
      id: 613,
      type: "SeargeImage2ImageAndInpainting",
      pos: [810, 380],
      size: {
        0: 320,
        1: 106,
      },
      flags: {
        pinned: true,
      },
      order: 33,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1568,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1569],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Image to Image and Inpainting",
      properties: {
        "Node name for S&R": "SeargeImage2ImageAndInpainting",
      },
      widgets_values: [0.5, 8, "masked - full"],
      color: "#323",
      bgcolor: "#535",
    },
    {
      id: 609,
      type: "SeargeModelSelector",
      pos: [1140, 620],
      size: {
        0: 520,
        1: 110,
      },
      flags: {
        pinned: true,
      },
      order: 38,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1572,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1573],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Model Selector",
      properties: {
        "Node name for S&R": "SeargeModelSelector",
      },
      widgets_values: [
        "sd_xl_base_1.0_0.9vae.safetensors",
        "sd_xl_refiner_1.0_0.9vae.safetensors",
        "sdxl_vae.safetensors",
      ],
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 468,
      type: "Note",
      pos: [1140, 940],
      size: {
        0: 520,
        1: 120,
      },
      flags: {
        pinned: true,
      },
      order: 8,
      mode: 0,
      title: "Workflow Information",
      properties: {
        text: "",
      },
      widgets_values: [
        "  Searge-SDXL: EVOLVED Workflow for SDXL 1.0  |  Version 4.3 updated on Oct 20, 2023\n\n    !!! REQUIRES CUSTOM NODE EXTENSION: https://github.com/SeargeDP/SeargeSDXL !!!\n\n More information about using this workflow is next to the upscaled image preview -=>\n                      ( 1.25 x 1.2 = 1.5  |  1.5 x 1.333 = 2.0 )",
      ],
      color: "#323",
      bgcolor: "#535",
    },
    {
      id: 619,
      type: "SeargeConditionMixing",
      pos: [2200, 860],
      size: {
        0: 520,
        1: 200,
      },
      flags: {
        pinned: true,
      },
      order: 40,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1685,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1686],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Condition Mixing - not implemented",
      properties: {
        "Node name for S&R": "SeargeConditionMixing",
      },
      color: "#322",
      bgcolor: "#533",
    },
    {
      id: 616,
      type: "SeargeCustomPromptMode",
      pos: [2200, 620],
      size: {
        0: 520,
        1: 200,
      },
      flags: {
        pinned: true,
      },
      order: 41,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1686,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1683],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Custom Prompt Mode - not implemented",
      properties: {
        "Node name for S&R": "SeargeCustomPromptMode",
      },
      color: "#332922",
      bgcolor: "#593930",
    },
    {
      id: 527,
      type: "SeargeMagicBox",
      pos: [-1300, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 56,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1624,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1626],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      title: "CLIP Conditioning",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["clip conditioning", "data stream", "data stream"],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 554,
      type: "SeargeCustomAfterVaeDecode",
      pos: [2300, 3430],
      size: {
        0: 300,
        1: 70,
      },
      flags: {
        pinned: true,
      },
      order: 66,
      mode: 0,
      inputs: [
        {
          name: "custom_output",
          type: "SRG_STAGE_OUTPUT",
          link: 1615,
        },
      ],
      outputs: [
        {
          name: "latent_image",
          type: "IMAGE",
          links: [1639],
          shape: 3,
          slot_index: 0,
          label: "image",
        },
      ],
      properties: {
        "Node name for S&R": "SeargeCustomAfterVaeDecode",
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 557,
      type: "Reroute",
      pos: [2600, 3550],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 68,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1639,
          pos: [37.5, 0],
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1476],
          label: "img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 558,
      type: "Reroute",
      pos: [2700, 3600],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 70,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1476,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1478],
          slot_index: 0,
          label: "img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 534,
      type: "SeargeMagicBox",
      pos: [1900, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 64,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1638,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1641],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: [1615],
          shape: 3,
          slot_index: 1,
        },
      ],
      title: "VAE Decode Sampled",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: [
        "vae decode sampled",
        "data stream",
        "custom stage & data stream",
      ],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 535,
      type: "SeargeMagicBox",
      pos: [1500, 3130],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 63,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1635,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_INPUT",
          link: null,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1638],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "custom_stage",
          type: "SRG_STAGE_OUTPUT",
          links: null,
          shape: 3,
        },
      ],
      title: "Latent Detailer",
      properties: {
        "Node name for S&R": "SeargeMagicBox",
      },
      widgets_values: ["latent detailer", "data stream", "data stream"],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 634,
      type: "SeargePreviewImage",
      pos: [2200, 40],
      size: {
        0: 520,
        1: 540,
      },
      flags: {
        pinned: true,
      },
      order: 110,
      mode: 0,
      inputs: [
        {
          name: "images",
          type: "IMAGE",
          link: 1596,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "images",
          type: "IMAGE",
          links: null,
          shape: 3,
        },
      ],
      title: "Upscaled Preview Image",
      properties: {
        "Node name for S&R": "SeargePreviewImage",
      },
      widgets_values: [true],
    },
    {
      id: 126,
      type: "Note",
      pos: [2730, 40],
      size: {
        0: 520,
        1: 540,
      },
      flags: {
        pinned: true,
      },
      order: 9,
      mode: 0,
      title: "Workflow Information",
      properties: {
        text: "",
      },
      widgets_values: [
        "\n Searge-SDXL v20231020.0815-beta-unstable-prerelease-0.007-sneak-preview-10\n\n [TODO: add description here]\n",
      ],
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 635,
      type: "Reroute",
      pos: [1650, -1050],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 108,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1595,
          pos: [37.5, 0],
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1596],
          label: "up img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 636,
      type: "Reroute",
      pos: [1500, -1200],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 106,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1598,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1595],
          label: "up img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 637,
      type: "Reroute",
      pos: [-5300, -1200],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 104,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1599,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1598],
          label: "up img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 638,
      type: "Reroute",
      pos: [-5800, -1600],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 99,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1601,
          slot_index: 0,
          pos: [37.5, 0],
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1600],
          label: "up img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 639,
      type: "Reroute",
      pos: [-5700, -1550],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 102,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1600,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1599],
          slot_index: 0,
          label: "up img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 640,
      type: "Reroute",
      pos: [-5800, 3500],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 95,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1602,
          slot_index: 0,
          pos: [37.5, 0],
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1601],
          slot_index: 0,
          label: "up img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 642,
      type: "Reroute",
      pos: [4600, 3700],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 86,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1613,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1603],
          slot_index: 0,
          label: "up img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 484,
      type: "SeargeSeparator",
      pos: [480, 520],
      size: {
        0: 650,
        1: 30,
      },
      flags: {
        pinned: true,
      },
      order: 10,
      mode: 0,
      title: "Advanced settings are below, commonly used settings are above",
      properties: {
        "Node name for S&R": "SeargeSeparator",
      },
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 614,
      type: "SeargeConditioningParameters",
      pos: [480, 590],
      size: {
        0: 320,
        1: 250,
      },
      flags: {
        pinned: true,
      },
      order: 21,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1564,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1565],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Conditioning Parameters",
      properties: {
        "Node name for S&R": "SeargeConditioningParameters",
      },
      widgets_values: [2, 1.5, 2, 1.5, 0.75, 6, 2.5, "gaussian", 0.1],
      color: "#232",
      bgcolor: "#353",
    },
    {
      id: 618,
      type: "SeargeHighResolution",
      pos: [810, 590],
      size: {
        0: 320,
        1: 250,
      },
      flags: {
        pinned: true,
      },
      order: 35,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1569,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1570],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "High Resolution",
      properties: {
        "Node name for S&R": "SeargeHighResolution",
      },
      widgets_values: [
        "none",
        "1.5x",
        0.2,
        0.1,
        0.5,
        0.05,
        0.1,
        "normal",
        "none",
      ],
      color: "#233",
      bgcolor: "#355",
    },
    {
      id: 615,
      type: "SeargeAdvancedParameters",
      pos: [480, 880],
      size: {
        0: 320,
        1: 180,
      },
      flags: {
        pinned: true,
      },
      order: 19,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1576,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1564],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Advanced Parameters (0 = disable)",
      properties: {
        "Node name for S&R": "SeargeAdvancedParameters",
      },
      widgets_values: ["rescale", 0.4, 0.2, 0.05, 0.1, "none"],
      color: "#2a363b",
      bgcolor: "#3f5159",
    },
    {
      id: 617,
      type: "SeargePromptStyles",
      pos: [810, 880],
      size: {
        0: 320,
        1: 180,
      },
      flags: {
        pinned: true,
      },
      order: 36,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1570,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1571],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Prompt Styles - not implemented",
      properties: {
        "Node name for S&R": "SeargePromptStyles",
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 522,
      type: "Reroute",
      pos: [-4400, 3050],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 49,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1420,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "SRG_DATA_STREAM",
          links: [1422],
          label: "data",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 521,
      type: "Reroute",
      pos: [4700, 3050],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 48,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1643,
        },
      ],
      outputs: [
        {
          name: "",
          type: "SRG_DATA_STREAM",
          links: [1420],
          label: "data",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 648,
      type: "Reroute",
      pos: [4600, 2900],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 47,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1642,
          pos: [37.5, 0],
        },
      ],
      outputs: [
        {
          name: "",
          type: "SRG_DATA_STREAM",
          links: [1643],
          label: "data",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 649,
      type: "Reroute",
      pos: [4500, 860],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 45,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1667,
        },
      ],
      outputs: [
        {
          name: "",
          type: "SRG_DATA_STREAM",
          links: [1645],
          slot_index: 0,
          label: "data",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 520,
      type: "Reroute",
      pos: [4600, 1000],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 46,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1645,
          pos: [37.5, 0],
        },
      ],
      outputs: [
        {
          name: "",
          type: "SRG_DATA_STREAM",
          links: [1642],
          slot_index: 0,
          label: "data",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 641,
      type: "Reroute",
      pos: [-6100, 3700],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 91,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1603,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1602],
          label: "up img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 562,
      type: "Reroute",
      pos: [-5600, -1200],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 80,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1473,
          slot_index: 0,
          pos: [37.5, 0],
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1487],
          label: "img",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 666,
      type: "SeargePreviewImage",
      pos: [390, 1200],
      size: {
        0: 300,
        1: 370,
      },
      flags: {
        pinned: true,
      },
      order: 23,
      mode: 0,
      inputs: [
        {
          name: "images",
          type: "IMAGE",
          link: 1671,
        },
      ],
      outputs: [
        {
          name: "images",
          type: "IMAGE",
          links: null,
          shape: 3,
        },
      ],
      title: "Preprocessor Preview",
      properties: {
        "Node name for S&R": "SeargePreviewImage",
      },
      widgets_values: [true],
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 658,
      type: "LoadImage",
      pos: [80, 1200],
      size: {
        0: 300,
        1: 370,
      },
      flags: {
        pinned: true,
      },
      order: 11,
      mode: 0,
      outputs: [
        {
          name: "IMAGE",
          type: "IMAGE",
          links: [1670],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "MASK",
          type: "MASK",
          links: null,
          shape: 3,
        },
      ],
      title: "Controlnet or Revision Source",
      properties: {
        "Node name for S&R": "LoadImage",
      },
      widgets_values: ["example.png", "image"],
      color: "#432",
      bgcolor: "#653",
    },
    {
      id: 668,
      type: "LoadImage",
      pos: [720, 1200],
      size: {
        0: 300,
        1: 370,
      },
      flags: {
        pinned: true,
      },
      order: 12,
      mode: 0,
      outputs: [
        {
          name: "IMAGE",
          type: "IMAGE",
          links: [1672],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "MASK",
          type: "MASK",
          links: null,
          shape: 3,
        },
      ],
      title: "Controlnet or Revision Source",
      properties: {
        "Node name for S&R": "LoadImage",
      },
      widgets_values: ["example.png", "image"],
      color: "#432",
      bgcolor: "#653",
    },
    {
      id: 667,
      type: "SeargePreviewImage",
      pos: [1030, 1200],
      size: {
        0: 300,
        1: 370,
      },
      flags: {
        pinned: true,
      },
      order: 26,
      mode: 0,
      inputs: [
        {
          name: "images",
          type: "IMAGE",
          link: 1673,
        },
      ],
      outputs: [
        {
          name: "images",
          type: "IMAGE",
          links: null,
          shape: 3,
        },
      ],
      title: "Preprocessor Preview",
      properties: {
        "Node name for S&R": "SeargePreviewImage",
      },
      widgets_values: [true],
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 669,
      type: "LoadImage",
      pos: [1360, 1200],
      size: {
        0: 300,
        1: 370,
      },
      flags: {
        pinned: true,
      },
      order: 13,
      mode: 0,
      outputs: [
        {
          name: "IMAGE",
          type: "IMAGE",
          links: [1674],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "MASK",
          type: "MASK",
          links: null,
          shape: 3,
        },
      ],
      title: "Controlnet or Revision Source",
      properties: {
        "Node name for S&R": "LoadImage",
      },
      widgets_values: ["example.png", "image"],
      color: "#432",
      bgcolor: "#653",
    },
    {
      id: 670,
      type: "SeargePreviewImage",
      pos: [1670, 1200],
      size: {
        0: 300,
        1: 370,
      },
      flags: {
        pinned: true,
      },
      order: 29,
      mode: 0,
      inputs: [
        {
          name: "images",
          type: "IMAGE",
          link: 1675,
        },
      ],
      outputs: [
        {
          name: "images",
          type: "IMAGE",
          links: null,
          shape: 3,
        },
      ],
      title: "Preprocessor Preview",
      properties: {
        "Node name for S&R": "SeargePreviewImage",
      },
      widgets_values: [true],
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 671,
      type: "LoadImage",
      pos: [2000, 1200],
      size: {
        0: 300,
        1: 370,
      },
      flags: {
        pinned: true,
      },
      order: 14,
      mode: 0,
      outputs: [
        {
          name: "IMAGE",
          type: "IMAGE",
          links: [1676],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "MASK",
          type: "MASK",
          links: null,
          shape: 3,
        },
      ],
      title: "Controlnet or Revision Source",
      properties: {
        "Node name for S&R": "LoadImage",
      },
      widgets_values: ["example.png", "image"],
      color: "#432",
      bgcolor: "#653",
    },
    {
      id: 672,
      type: "SeargePreviewImage",
      pos: [2310, 1200],
      size: {
        0: 300,
        1: 370,
      },
      flags: {
        pinned: true,
      },
      order: 32,
      mode: 0,
      inputs: [
        {
          name: "images",
          type: "IMAGE",
          link: 1677,
        },
      ],
      outputs: [
        {
          name: "images",
          type: "IMAGE",
          links: null,
          shape: 3,
        },
      ],
      title: "Preprocessor Preview",
      properties: {
        "Node name for S&R": "SeargePreviewImage",
      },
      widgets_values: [true],
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 673,
      type: "LoadImage",
      pos: [2640, 1200],
      size: {
        0: 300,
        1: 370,
      },
      flags: {
        pinned: true,
      },
      order: 15,
      mode: 0,
      outputs: [
        {
          name: "IMAGE",
          type: "IMAGE",
          links: [1678],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "MASK",
          type: "MASK",
          links: null,
          shape: 3,
        },
      ],
      title: "Controlnet or Revision Source",
      properties: {
        "Node name for S&R": "LoadImage",
      },
      widgets_values: ["example.png", "image"],
      color: "#432",
      bgcolor: "#653",
    },
    {
      id: 647,
      type: "LoadImageMask",
      pos: [-450, 1100],
      size: {
        0: 400,
        1: 470,
      },
      flags: {
        pinned: true,
      },
      order: 16,
      mode: 0,
      outputs: [
        {
          name: "MASK",
          type: "MASK",
          links: [1660],
          shape: 3,
        },
      ],
      title: "Upload Inpainting Mask (as grayscale image)",
      properties: {
        "Node name for S&R": "LoadImageMask",
      },
      widgets_values: ["example.png", "green", "image"],
      color: "#233",
      bgcolor: "#355",
    },
    {
      id: 674,
      type: "SeargePreviewImage",
      pos: [2950, 1200],
      size: {
        0: 300,
        1: 370,
      },
      flags: {
        pinned: true,
      },
      order: 34,
      mode: 0,
      inputs: [
        {
          name: "images",
          type: "IMAGE",
          link: 1679,
        },
      ],
      outputs: [
        {
          name: "images",
          type: "IMAGE",
          links: null,
          shape: 3,
        },
      ],
      title: "Preprocessor Preview",
      properties: {
        "Node name for S&R": "SeargePreviewImage",
      },
      widgets_values: [true],
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 676,
      type: "SeargeCustomAfterUpscaling",
      pos: [4300, 3430],
      size: {
        0: 300,
        1: 70,
      },
      flags: {
        pinned: true,
      },
      order: 78,
      mode: 0,
      inputs: [
        {
          name: "custom_output",
          type: "SRG_STAGE_OUTPUT",
          link: 1658,
        },
      ],
      outputs: [
        {
          name: "image",
          type: "IMAGE",
          links: [1659],
          shape: 3,
          slot_index: 0,
        },
      ],
      properties: {
        "Node name for S&R": "SeargeCustomAfterUpscaling",
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 620,
      type: "SeargeUpscaleModels",
      pos: [1140, 770],
      size: {
        0: 520,
        1: 130,
      },
      flags: {
        pinned: true,
      },
      order: 37,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1571,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1572],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Upscale Models Selector",
      properties: {
        "Node name for S&R": "SeargeUpscaleModels",
      },
      widgets_values: [
        "1x-ITF-SkinDiffDetail-Lite-v1.pth",
        "4x_Nickelback_70000G.pth",
        "4x-UltraSharp.pth",
        "4x_NMKD-Siax_200k.pth",
      ],
      color: "#432",
      bgcolor: "#653",
    },
    {
      id: 621,
      type: "SeargeLoras",
      pos: [1670, 770],
      size: {
        0: 520,
        1: 290,
      },
      flags: {
        pinned: true,
      },
      order: 39,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1573,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1685],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Lora Selector",
      properties: {
        "Node name for S&R": "SeargeLoras",
      },
      widgets_values: [
        "sd_xl_offset_example-lora_1.0.safetensors",
        0.2,
        "none",
        0.5,
        "none",
        0.5,
        "none",
        0.5,
        "none",
        0.5,
      ],
      color: "#2a363b",
      bgcolor: "#3f5159",
    },
    {
      id: 523,
      type: "Reroute",
      pos: [-4300, 3130],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 50,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1422,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "SRG_DATA_STREAM",
          links: [1665],
          slot_index: 0,
          label: "data",
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 628,
      type: "SeargePipelineStart",
      pos: [4100, 860],
      size: {
        0: 300,
        1: 140,
      },
      flags: {
        pinned: true,
      },
      order: 44,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1681,
        },
        {
          name: "additional_data",
          type: "SRG_DATA_STREAM",
          link: 1668,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1667],
          shape: 3,
          slot_index: 0,
        },
      ],
      properties: {
        "Node name for S&R": "SeargePipelineStart",
      },
      widgets_values: ["4.3"],
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 626,
      type: "SeargeImageAdapterV2",
      pos: [-400, 1630],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 18,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: null,
        },
        {
          name: "source_image",
          type: "IMAGE",
          link: 1587,
          slot_index: 1,
        },
        {
          name: "image_mask",
          type: "MASK",
          link: 1588,
          slot_index: 2,
        },
        {
          name: "uploaded_mask",
          type: "MASK",
          link: 1660,
          slot_index: 3,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1669],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "image_inputs",
          type: "SRG_DATA_STREAM",
          links: null,
          shape: 3,
        },
      ],
      title: "Image Adapter",
      properties: {
        "Node name for S&R": "SeargeImageAdapterV2",
      },
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 622,
      type: "SeargePromptAdapterV2",
      pos: [-400, -270],
      size: {
        0: 300,
        1: 170,
      },
      flags: {
        pinned: true,
      },
      order: 17,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: null,
          slot_index: 0,
        },
        {
          name: "main_prompt",
          type: "SRG_PROMPT_TEXT",
          link: 1577,
        },
        {
          name: "secondary_prompt",
          type: "SRG_PROMPT_TEXT",
          link: 1578,
        },
        {
          name: "style_prompt",
          type: "SRG_PROMPT_TEXT",
          link: 1579,
        },
        {
          name: "negative_main_prompt",
          type: "SRG_PROMPT_TEXT",
          link: 1580,
        },
        {
          name: "negative_secondary_prompt",
          type: "SRG_PROMPT_TEXT",
          link: 1581,
        },
        {
          name: "negative_style_prompt",
          type: "SRG_PROMPT_TEXT",
          link: 1582,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1576],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "prompts",
          type: "SRG_DATA_STREAM",
          links: null,
          shape: 3,
        },
      ],
      title: "Prompt Adapter",
      properties: {
        "Node name for S&R": "SeargePromptAdapterV2",
      },
      color: "#222",
      bgcolor: "#000",
    },
    {
      id: 683,
      type: "SeargeControlnetModels",
      pos: [2730, 860],
      size: {
        0: 520,
        1: 200,
      },
      flags: {
        pinned: true,
      },
      order: 43,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1687,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1681],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "Controlnet Models Selector",
      properties: {
        "Node name for S&R": "SeargeControlnetModels",
      },
      widgets_values: [
        "clip_vision_g.safetensors",
        "control-lora-canny-rank256.safetensors",
        "control-lora-depth-rank256.safetensors",
        "control-lora-recolor-rank256.safetensors",
        "control-lora-sketch-rank256.safetensors",
        "none",
      ],
      color: "#323",
      bgcolor: "#535",
    },
    {
      id: 678,
      type: "SeargeControlnetAdapterV2",
      pos: [230, 1630],
      size: {
        0: 310,
        1: 270,
      },
      flags: {
        pinned: true,
      },
      order: 20,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1669,
        },
        {
          name: "source_image",
          type: "IMAGE",
          link: 1670,
          slot_index: 1,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1664],
          shape: 3,
        },
        {
          name: "preview",
          type: "IMAGE",
          links: [1671],
          shape: 3,
          slot_index: 1,
        },
      ],
      title: "Controlnet and Revision Adapter",
      properties: {
        "Node name for S&R": "SeargeControlnetAdapterV2",
      },
      widgets_values: ["none", true, 1, 0, 1, 0, 1, 0, false],
      color: "#323",
      bgcolor: "#535",
    },
    {
      id: 679,
      type: "SeargeControlnetAdapterV2",
      pos: [870, 1630],
      size: {
        0: 310,
        1: 270,
      },
      flags: {
        pinned: true,
      },
      order: 22,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1664,
          slot_index: 0,
        },
        {
          name: "source_image",
          type: "IMAGE",
          link: 1672,
          slot_index: 1,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1663],
          shape: 3,
        },
        {
          name: "preview",
          type: "IMAGE",
          links: [1673],
          shape: 3,
          slot_index: 1,
        },
      ],
      title: "Controlnet and Revision Adapter",
      properties: {
        "Node name for S&R": "SeargeControlnetAdapterV2",
      },
      widgets_values: ["none", true, 1, 0, 1, 0, 1, 0, false],
      color: "#323",
      bgcolor: "#535",
    },
    {
      id: 680,
      type: "SeargeControlnetAdapterV2",
      pos: [1510, 1630],
      size: {
        0: 310,
        1: 270,
      },
      flags: {
        pinned: true,
      },
      order: 25,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1663,
          slot_index: 0,
        },
        {
          name: "source_image",
          type: "IMAGE",
          link: 1674,
          slot_index: 1,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1662],
          shape: 3,
        },
        {
          name: "preview",
          type: "IMAGE",
          links: [1675],
          shape: 3,
          slot_index: 1,
        },
      ],
      title: "Controlnet and Revision Adapter",
      properties: {
        "Node name for S&R": "SeargeControlnetAdapterV2",
      },
      widgets_values: ["none", true, 1, 0, 1, 0, 1, 0, false],
      color: "#323",
      bgcolor: "#535",
    },
    {
      id: 681,
      type: "SeargeControlnetAdapterV2",
      pos: [2150, 1630],
      size: {
        0: 310,
        1: 270,
      },
      flags: {
        pinned: true,
      },
      order: 28,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1662,
          slot_index: 0,
        },
        {
          name: "source_image",
          type: "IMAGE",
          link: 1676,
          slot_index: 1,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1661],
          shape: 3,
        },
        {
          name: "preview",
          type: "IMAGE",
          links: [1677],
          shape: 3,
          slot_index: 1,
        },
      ],
      title: "Controlnet and Revision Adapter",
      properties: {
        "Node name for S&R": "SeargeControlnetAdapterV2",
      },
      widgets_values: ["none", true, 1, 0, 1, 0, 1, 0, false],
      color: "#323",
      bgcolor: "#535",
    },
    {
      id: 682,
      type: "SeargeControlnetAdapterV2",
      pos: [2790, 1630],
      size: {
        0: 310,
        1: 270,
      },
      flags: {
        pinned: true,
      },
      order: 31,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1661,
          slot_index: 0,
        },
        {
          name: "source_image",
          type: "IMAGE",
          link: 1678,
          slot_index: 1,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1668],
          shape: 3,
          slot_index: 0,
        },
        {
          name: "preview",
          type: "IMAGE",
          links: [1679],
          shape: 3,
          slot_index: 1,
        },
      ],
      title: "Controlnet and Revision Adapter",
      properties: {
        "Node name for S&R": "SeargeControlnetAdapterV2",
      },
      widgets_values: ["none", true, 1, 0, 1, 0, 1, 0, false],
      color: "#323",
      bgcolor: "#535",
    },
    {
      id: 573,
      type: "SeargeCustomAfterVaeDecode",
      pos: [3500, 3430],
      size: {
        0: 300,
        1: 70,
      },
      flags: {
        pinned: true,
      },
      order: 72,
      mode: 0,
      inputs: [
        {
          name: "custom_output",
          type: "SRG_STAGE_OUTPUT",
          link: 1682,
        },
      ],
      outputs: [
        {
          name: "latent_image",
          type: "IMAGE",
          links: [1646],
          shape: 3,
          slot_index: 0,
          label: "image",
        },
      ],
      properties: {
        "Node name for S&R": "SeargeCustomAfterVaeDecode",
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 645,
      type: "Reroute",
      pos: [3800, 3540],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 75,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1646,
          slot_index: 0,
          pos: [37.5, 0],
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1612],
          label: "hr img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: true,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 563,
      type: "Reroute",
      pos: [3800, 3650],
      size: [75, 26],
      flags: {
        pinned: true,
      },
      order: 79,
      mode: 0,
      inputs: [
        {
          name: "",
          type: "*",
          link: 1612,
          slot_index: 0,
        },
      ],
      outputs: [
        {
          name: "",
          type: "IMAGE",
          links: [1480],
          label: "hr img",
          slot_index: 0,
        },
      ],
      properties: {
        showOutputText: false,
        horizontal: false,
      },
      color: "#223",
      bgcolor: "#335",
    },
    {
      id: 684,
      type: "SeargeFreeU",
      pos: [2730, 620],
      size: {
        0: 520,
        1: 200,
      },
      flags: {
        pinned: true,
      },
      order: 42,
      mode: 0,
      inputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          link: 1683,
        },
      ],
      outputs: [
        {
          name: "data",
          type: "SRG_DATA_STREAM",
          links: [1687],
          shape: 3,
          slot_index: 0,
        },
      ],
      title: "FreeU",
      properties: {
        "Node name for S&R": "SeargeFreeU",
      },
      widgets_values: ["custom", 1, 1.2, 1, 1, "freeu_v2"],
      color: "#223",
      bgcolor: "#335",
    },
  ],
  links: [
    [424, 98, 0, 130, 6, "FLOAT"],
    [1399, 498, 0, 19, 0, "IMAGE"],
    [1400, 499, 0, 463, 0, "IMAGE"],
    [1401, 497, 0, 499, 0, "*"],
    [1402, 500, 0, 498, 0, "*"],
    [1404, 502, 0, 87, 0, "IMAGE"],
    [1407, 503, 0, 497, 0, "*"],
    [1420, 521, 0, 522, 0, "*"],
    [1422, 522, 0, 523, 0, "*"],
    [1458, 544, 0, 551, 0, "*"],
    [1472, 559, 0, 560, 0, "*"],
    [1473, 560, 0, 562, 0, "*"],
    [1476, 557, 0, 558, 0, "*"],
    [1478, 558, 0, 559, 0, "*"],
    [1480, 563, 0, 564, 0, "*"],
    [1481, 564, 0, 565, 0, "*"],
    [1482, 565, 0, 566, 0, "*"],
    [1483, 567, 0, 502, 0, "*"],
    [1484, 568, 0, 503, 0, "*"],
    [1487, 562, 0, 567, 0, "*"],
    [1488, 566, 0, 569, 0, "*"],
    [1489, 569, 0, 568, 0, "*"],
    [1564, 615, 0, 614, 0, "SRG_DATA_STREAM"],
    [1565, 614, 0, 611, 0, "SRG_DATA_STREAM"],
    [1566, 611, 0, 610, 0, "SRG_DATA_STREAM"],
    [1567, 610, 0, 612, 0, "SRG_DATA_STREAM"],
    [1568, 612, 0, 613, 0, "SRG_DATA_STREAM"],
    [1569, 613, 0, 618, 0, "SRG_DATA_STREAM"],
    [1570, 618, 0, 617, 0, "SRG_DATA_STREAM"],
    [1571, 617, 0, 620, 0, "SRG_DATA_STREAM"],
    [1572, 620, 0, 609, 0, "SRG_DATA_STREAM"],
    [1573, 609, 0, 621, 0, "SRG_DATA_STREAM"],
    [1576, 622, 0, 615, 0, "SRG_DATA_STREAM"],
    [1577, 3, 0, 622, 1, "SRG_PROMPT_TEXT"],
    [1578, 5, 0, 622, 2, "SRG_PROMPT_TEXT"],
    [1579, 6, 0, 622, 3, "SRG_PROMPT_TEXT"],
    [1580, 7, 0, 622, 4, "SRG_PROMPT_TEXT"],
    [1581, 8, 0, 622, 5, "SRG_PROMPT_TEXT"],
    [1582, 8, 0, 622, 6, "SRG_PROMPT_TEXT"],
    [1587, 268, 0, 626, 1, "IMAGE"],
    [1588, 269, 1, 626, 2, "MASK"],
    [1594, 87, 0, 86, 0, "IMAGE"],
    [1595, 636, 0, 635, 0, "*"],
    [1596, 635, 0, 634, 0, "IMAGE"],
    [1597, 502, 0, 500, 0, "*"],
    [1598, 637, 0, 636, 0, "*"],
    [1599, 639, 0, 637, 0, "*"],
    [1600, 638, 0, 639, 0, "*"],
    [1601, 640, 0, 638, 0, "*"],
    [1602, 641, 0, 640, 0, "*"],
    [1603, 642, 0, 641, 0, "*"],
    [1612, 645, 0, 563, 0, "*"],
    [1613, 571, 0, 642, 0, "*"],
    [1615, 534, 1, 554, 0, "SRG_STAGE_OUTPUT"],
    [1619, 537, 0, 506, 0, "SRG_DATA_STREAM"],
    [1620, 506, 0, 524, 0, "SRG_DATA_STREAM"],
    [1622, 524, 0, 525, 0, "SRG_DATA_STREAM"],
    [1623, 525, 0, 526, 0, "SRG_DATA_STREAM"],
    [1624, 526, 0, 527, 0, "SRG_DATA_STREAM"],
    [1626, 527, 0, 528, 0, "SRG_DATA_STREAM"],
    [1627, 528, 0, 529, 0, "SRG_DATA_STREAM"],
    [1628, 529, 0, 530, 0, "SRG_DATA_STREAM"],
    [1629, 530, 0, 531, 0, "SRG_DATA_STREAM"],
    [1631, 531, 0, 532, 0, "SRG_DATA_STREAM"],
    [1632, 532, 0, 533, 0, "SRG_DATA_STREAM"],
    [1635, 533, 0, 535, 0, "SRG_DATA_STREAM"],
    [1638, 535, 0, 534, 0, "SRG_DATA_STREAM"],
    [1639, 554, 0, 557, 0, "*"],
    [1641, 534, 0, 536, 0, "SRG_DATA_STREAM"],
    [1642, 520, 0, 648, 0, "*"],
    [1643, 648, 0, 521, 0, "*"],
    [1645, 649, 0, 520, 0, "*"],
    [1646, 573, 0, 645, 0, "*"],
    [1647, 536, 0, 538, 0, "SRG_DATA_STREAM"],
    [1650, 540, 0, 541, 0, "SRG_DATA_STREAM"],
    [1651, 539, 0, 540, 0, "SRG_DATA_STREAM"],
    [1653, 538, 0, 539, 0, "SRG_DATA_STREAM"],
    [1654, 541, 0, 542, 0, "SRG_DATA_STREAM"],
    [1655, 542, 0, 543, 0, "SRG_DATA_STREAM"],
    [1656, 543, 0, 544, 0, "SRG_DATA_STREAM"],
    [1657, 544, 0, 590, 0, "SRG_DATA_STREAM"],
    [1658, 541, 1, 676, 0, "SRG_STAGE_OUTPUT"],
    [1659, 676, 0, 571, 0, "*"],
    [1660, 647, 0, 626, 3, "MASK"],
    [1661, 681, 0, 682, 0, "SRG_DATA_STREAM"],
    [1662, 680, 0, 681, 0, "SRG_DATA_STREAM"],
    [1663, 679, 0, 680, 0, "SRG_DATA_STREAM"],
    [1664, 678, 0, 679, 0, "SRG_DATA_STREAM"],
    [1665, 523, 0, 537, 0, "SRG_DATA_STREAM"],
    [1667, 628, 0, 649, 0, "*"],
    [1668, 682, 0, 628, 1, "SRG_DATA_STREAM"],
    [1669, 626, 0, 678, 0, "SRG_DATA_STREAM"],
    [1670, 658, 0, 678, 1, "IMAGE"],
    [1671, 678, 1, 666, 0, "IMAGE"],
    [1672, 668, 0, 679, 1, "IMAGE"],
    [1673, 679, 1, 667, 0, "IMAGE"],
    [1674, 669, 0, 680, 1, "IMAGE"],
    [1675, 680, 1, 670, 0, "IMAGE"],
    [1676, 671, 0, 681, 1, "IMAGE"],
    [1677, 681, 1, 672, 0, "IMAGE"],
    [1678, 673, 0, 682, 1, "IMAGE"],
    [1679, 682, 1, 674, 0, "IMAGE"],
    [1681, 683, 0, 628, 0, "SRG_DATA_STREAM"],
    [1682, 539, 1, 573, 0, "SRG_STAGE_OUTPUT"],
    [1683, 616, 0, 684, 0, "SRG_DATA_STREAM"],
    [1685, 621, 0, 619, 0, "SRG_DATA_STREAM"],
    [1686, 619, 0, 616, 0, "SRG_DATA_STREAM"],
    [1687, 684, 0, 683, 0, "SRG_DATA_STREAM"],
  ],
  groups: [],
  config: {},
  extra: {},
  version: 0.4,
};
