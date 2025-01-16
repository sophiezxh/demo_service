
ETTS_data = [
 {'text': 'One two three, where is your breakfast.',
  'emotion': 'Happy',
  'origin_audio_path': './static/demo_audio/origin/0012_000986.wav',
  'vits_label_path': './static/demo_audio/vits_label/0012_000986.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0012_000986.wav'},
 
 {'text': 'The nastiest things they saw were the cobwebs.',
  'emotion': 'Happy',
  'origin_audio_path': './static/demo_audio/origin/0015_000745.wav',
  'vits_label_path': './static/demo_audio/vits_label/0015_000745.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0015_000745.wav'},
 
 {'text': 'Ask god to help you.',
  'emotion': 'Happy',
  'origin_audio_path': './static/demo_audio/origin/0019_000922.wav',
  'vits_label_path': './static/demo_audio/vits_label/0019_000922.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0019_000922.wav'},
 
 {'text': 'Will call her Lily, for short.',
  'emotion': 'Surprise',
  'origin_audio_path': './static/demo_audio/origin/0020_001632.wav',
  'vits_label_path': './static/demo_audio/vits_label/0020_001632.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0020_001632.wav'},
 
 {'text': 'I blinked my eyes hard.',
  'emotion': 'Surprise',
  'origin_audio_path': './static/demo_audio/origin/0015_001535.wav',
  'vits_label_path': './static/demo_audio/vits_label/0015_001535.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0015_001535.wav'},
 
 {'text': 'I know you .',
  'emotion': 'Sad',
  'origin_audio_path': './static/demo_audio/origin/0016_001328.wav',
  'vits_label_path': './static/demo_audio/vits_label/0016_001328.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0016_001328.wav'},
  
 {'text': 'They were children of mine.',
  'emotion': 'Sad',
  'origin_audio_path': './static/demo_audio/origin/0016_001167.wav',
  'vits_label_path': './static/demo_audio/vits_label/0016_001167.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0016_001167.wav'},
  
 {'text': "I've hit the wrong nose.",
  'emotion': 'Neutral',
  'origin_audio_path': './static/demo_audio/origin/0012_000099.wav',
  'vits_label_path': './static/demo_audio/vits_label/0012_000099.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0012_000099.wav'},
 
 {'text': 'She said in subdued voice.',
  'emotion': 'Neutral',
  'origin_audio_path': './static/demo_audio/origin/0017_000263.wav',
  'vits_label_path': './static/demo_audio/vits_label/0017_000263.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0017_000263.wav'},
   
 {'text': 'Because he was a man with infinite resource and sagacity.',
  'emotion': 'Neutral',
  'origin_audio_path': './static/demo_audio/origin/0017_000274.wav',
  'vits_label_path': './static/demo_audio/vits_label/0017_000274.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0017_000274.wav'},
 
 {'text': 'Enough,you a foolish chatter.',
  'emotion': 'Neutral',
  'origin_audio_path': './static/demo_audio/origin/0018_000341.wav',
  'vits_label_path': './static/demo_audio/vits_label/0018_000341.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0018_000341.wav'},
 
 {'text': 'Tom now let our arrows fly!',
  'emotion': 'Angry',
  'origin_audio_path': './static/demo_audio/origin/0019_000420.wav',
  'vits_label_path': './static/demo_audio/vits_label/0019_000420.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0019_000420.wav'},
 ]


emotionColor = {
    'Neutral': '#a0a0a0',
    'Joy': '#ffbf7f',
    'Sadness': '#8fb2ff',
    'Surprise': '#b266ff',
    'Disgust': '#29a329',
    'Happy': '#ffbf7f',
    'Sad': '#8fb2ff',
    'Angry': '#ff4c4c',

    'Happily': '#ffbf7f',
    'Sadly': '#8fb2ff',
    'Fearful': '#ff7f00',
    'Fearfully': '#ff7f00',
    'Angry': '#ff4c4c',
    'Angrily': '#ff4c4c',
    'Disgusted': '#29a329',
    'Disgustingly': '#29a329',
    'Surprised': '#b266ff',
}

function createETTSTable(data) {
    var table = document.getElementById("ETTS_table").getElementsByTagName("tbody")[0];
    for (var i = 0; i < data.length; i++){
        var row = table.insertRow();
        var textCell = row.insertCell(0);
        var emotionCell = row.insertCell(1);
        var originAudioCell = row.insertCell(2);
        var vitsLabelCell = row.insertCell(3);
        var mmttsVitsCell = row.insertCell(4);

        var text = document.createTextNode(data[i].text);
        textCell.appendChild(text);
      
        var emotion = document.createTextNode(data[i].emotion);
        emotionCell.appendChild(emotion);
        emotionCell.style.color = emotionColor[data[i].emotion];

        var originAudio = document.createElement("audio");
        originAudio.src = data[i].origin_audio_path;
        originAudio.controls = true;
        originAudioCell.appendChild(originAudio);

        var vitsLabel = document.createElement("audio");
        vitsLabel.src = data[i].vits_label_path;
        vitsLabel.controls = true;
        vitsLabelCell.appendChild(vitsLabel);

        var mmttsVits = document.createElement("audio");
        mmttsVits.src = data[i].mmtts_vits_path;
        mmttsVits.controls = true;
        mmttsVitsCell.appendChild(mmttsVits);


    }
}

window.onload = function(){
    createETTSTable(ETTS_data)
}
