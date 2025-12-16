// Initiate the word count to 0
document.getElementById('textarea').addEventListener('input', function() {
    const text = this.value.trim();
    const wordCount = text ? text.split(/\s+/).length : 0;
    document.getElementById('wordCount').textContent = wordCount;
    if (wordCount > 500) {
      document.getElementsByClassName('number')[0].classList.add("text-red-600");
      document.getElementsByClassName('number')[1].classList.add("text-red-600");
    }else{
      document.getElementsByClassName('number')[0].classList.remove("text-red-600");
      document.getElementsByClassName('number')[1].classList.remove("text-red-600");
    }
});


function validateWordCount(event) {
  const wordCount = parseInt(document.getElementById('wordCount').textContent);
  const errorMessage = document.getElementById('errorMessage');
  
  if (wordCount > 500 || wordCount < 50) {
      event.preventDefault();
      errorMessage.classList.remove('hidden');
      return false;
  }
  errorMessage.classList.add('hidden');
  document.getElementById("predictForm").submit();
}



